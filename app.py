import streamlit as st
from PyPDF2 import PdfReader
import boto3
import pandas as pd
from io import BytesIO
import json

# Set page configuration with custom title and favicon
st.set_page_config(
    page_title="Resume Scorer",
    page_icon="NIWC_Atlantic_Logo.jpg"
)

# AWS S3 Configuration
BUCKET_NAME = 'resume-scorer-mockup-bucket'
EXCEL_FILE_KEY = 'sample_excel.xlsx'

# Function to download Excel file from S3
def download_excel_from_s3(bucket_name, file_key):
    s3_client = boto3.client('s3', region_name='us-east-1')
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        file_content = response['Body'].read()
        return pd.read_excel(BytesIO(file_content))
    except Exception as e:
        st.error(f"Error downloading Excel file: {str(e)}")
        return None

# Function to upload Excel file to S3
def upload_excel_to_s3(df, bucket_name, file_key):
    s3_client = boto3.client('s3', region_name='us-east-1')
    try:
        with BytesIO() as buffer:
            df.to_excel(buffer, index=False)
            buffer.seek(0)
            s3_client.upload_fileobj(buffer, bucket_name, file_key)
        st.success("Excel file successfully uploaded to S3.")
    except Exception as e:
        st.error(f"Error uploading Excel file to S3: {str(e)}")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    all_text = ""
    for page in pdf_reader.pages:
        all_text += page.extract_text()
    return all_text

# Function to analyze text using AWS Bedrock LLM with structured prompt
def analyze_text_with_bedrock(text):
    client = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    model_id = 'amazon.titan-text-premier-v1:0'
    
    # Define the prompt with structured sections
    grading_prompt = (
        "Human: Please evaluate the content based on the following grading criteria:\n\n"
        
        "GPA:\n"
        "  - 3.0 or greater = Above Average (5 points)\n"
        "  - 2.51-2.99 = Average (3 points)\n"
        "  - 2.50 or below = Below Average (0 points)\n\n"
        
        "School Activities:\n"
        "  - 2 or more activities = Above Average (5 points)\n"
        "  - 1 activity = Average (3 points)\n"
        "  - 0 activities = Below Average (0 points)\n\n"
        
        "Awards:\n"
        "  - 2 or more awards = Above Average (5 points)\n"
        "  - 1 award = Average (3 points)\n"
        "  - 0 awards = Below Average (0 points)\n\n"
        
        "Volunteer Service:\n"
        "  - 2 or more years of service = Above Average (5 points)\n"
        "  - 1 to 2 (not including 2) years of service = Average (3 points)\n"
        "  - 0-1 years of service = Below Average (0 points)\n\n"
        
        "Leadership:\n"
        "  - President or VP of something = Above Average (5 points)\n"
        "  - Other leadership position = Average (3 points)\n"
        "  - None = Below Average (0 points)\n\n"
        
        "Total Score: Sum of all points based on the above criteria.\n\n"
        
        "Please format the response as follows:\n\n"
        
        "### Personal Information ###\n"
        "Name: [Extracted Name]\n"
        "Email: [Extracted Email]\n"
        "Phone Number: [Extracted Phone Number]\n\n"
        
        "### Grading ###\n"
        "GPA: [Extracted GPA Value, e.g., 3.7] - [Corresponding Points]\n"
        "School Activities: [Number of Activities, e.g., 2] - [Corresponding Points]\n"
        "Awards: [Number of Awards, e.g., 1] - [Corresponding Points]\n"
        "Volunteer Service: [Years of Service, e.g., 1.5 years] - [Corresponding Points]\n"
        "Leadership: [Position Held, e.g., President] - [Corresponding Points]\n"
        "Total Score: [Total Points]\n\n"
        
        "### Explanation ###\n"
        "[Provide a detailed explanation of how the total score was calculated based on the above criteria.]\n\n"
        
        "Content to evaluate:\n"
        f"{text}\n"
        "Assistant:"
    )


    
    # Payload for the model
    payload = {
        'inputText': grading_prompt
    }
    request_body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 50000,
        "temperature": 0.7,
        "messages": [
            {"role": "user", "content": grading_prompt}
        ]
    })
    
    # Invoke the Bedrock model
    response = client.invoke_model(
        modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
        contentType='application/json',
        accept='application/json',
        body=request_body
    )
    
    # Read and decode the response
    response_body = response['body'].read().decode('utf-8')
    return json.loads(response_body)

# Function to parse the model response based on sections
def parse_model_response(response):
    # Check if 'content' is in the response
    if 'content' in response and isinstance(response['content'], list):
        # Get the text content from the first item in the list
        output_text = response['content'][0].get('text', 'No output text found.')
        
        # Split the output text into different sections
        sections = {'Personal Information': '', 'Grading': '', 'Explanation': ''}
        current_section = None
        for line in output_text.splitlines():
            if line.startswith("###"):
                current_section = line.strip().replace("###", "").strip()
            elif current_section and line.strip():
                sections[current_section] += line.strip() + "\n"
        
        # Extract individual fields from the sections
        personal_info = sections.get('Personal Information', '')
        grading_info = sections.get('Grading', '')
        explanation_info = sections.get('Explanation', '')

        extracted_info = {
            'Name': extract_field(personal_info, 'Name'),
            'Email': extract_field(personal_info, 'Email'),
            'Phone Number': extract_field(personal_info, 'Phone Number'),
            'GPA': extract_field(grading_info, 'GPA'),
            'School Activities': extract_field(grading_info, 'School Activities'),
            'Awards': extract_field(grading_info, 'Awards'),
            'Volunteer Service': extract_field(grading_info, 'Volunteer Service'),
            'Leadership': extract_field(grading_info, 'Leadership'),
            'Scores Explanation': explanation_info.strip()
        }
        return extracted_info
    else:
        return None

# Function to extract field value from section text
def extract_field(text, field_name):
    try:
        start_idx = text.index(f"{field_name}:") + len(f"{field_name}:")
        end_idx = text.index('\n', start_idx)
        return text[start_idx:end_idx].strip()
    except:
        return 'Not found'

# Streamlit UI
st.title("NIWC-A Resume Scorer Mockup")
st.write("Upload a PDF file to extract and grade its content based on the criteria.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Check if a file has been uploaded
if uploaded_file is not None:
    # Extract text from the PDF file
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.subheader("Extracted Text:")
    st.text_area("PDF Text", pdf_text, height=200)

    # Analyze the text using the LLM model
    if st.button("Analyze and Grade"):
        with st.spinner("Analyzing with AWS Bedrock..."):
            try:
                # Analyze with the grading prompt
                result = analyze_text_with_bedrock(pdf_text)
                
                # Display the full JSON response on the Streamlit page
                st.subheader("Full JSON Response:")
                st.json(result)
                
                # Parse the response for required fields
                extracted_info = parse_model_response(result)
                if extracted_info:
                    st.subheader("Extracted Information:")
                    st.write(extracted_info)
                    
                    # Download Excel file from S3
                    df = download_excel_from_s3(BUCKET_NAME, EXCEL_FILE_KEY)
                    if df is not None:
                        # Create a DataFrame for the new row
                        new_row_df = pd.DataFrame([extracted_info])
                        df = pd.concat([df, new_row_df], ignore_index=True) 
                        
                        # Upload updated Excel file back to S3
                        upload_excel_to_s3(df, BUCKET_NAME, EXCEL_FILE_KEY)
                else:
                    st.warning("Could not extract necessary information from the model response.")
                
            except Exception as e:
                st.error(f"Error analyzing the PDF: {str(e)}")
