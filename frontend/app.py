import streamlit as st
import requests

# Page configuration with a professional theme
st.set_page_config(
    page_title="Wayne Manor Intelligence", 
    page_icon="", 
    layout="centered"
)

st.title("Alfred: Multi-Agent Document Intelligence")
st.markdown("""
*Head Butler & Intelligence Orchestrator at your service.*
Currently supporting: **PDF, Images, Word (DOCX), Excel, CSV, and Text.**
---
""")

# SIDEBAR: Project Status
with st.sidebar:
    st.header("System Status")
    st.success("Supervisor: Online")
    st.info("Resume Specialist: Active")
    st.info("Document Analyst: Active")
    if st.button("Clear Briefing History"):
        st.rerun()

# FILE UPLOADER: Expanded types for Multi-Agent support
uploaded_file = st.file_uploader(
    "Upload Master Wayne's notes/reports", 
    type=["png", "jpg", "jpeg", "pdf", "txt", "csv", "docx", "xlsx"]
)

# CHAT INTERFACE
user_input = st.text_input("How can I assist you, sir/madam?", placeholder="e.g., Analyze this resume and extract hard skills.")

if st.button("Consult Alfred"):
    if not user_input:
        st.warning("I require instructions to proceed, sir.")
    else:
        # User message display
        st.chat_message("user", avatar="👤").write(user_input)
        
        with st.spinner("Alfred is delegating to specialized agents..."):
            try:
                # Backend URL
                url = "http://localhost:8000/chat"
                
                # Payload construction (multipart/form-data)
                payload = {"message": user_input}
                
                files = None
                if uploaded_file:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Request to FastAPI
                response = requests.post(url, data=payload, files=files)
                
                if response.status_code == 200:
                    answer = response.json().get("response")
                    
                    # Custom CSS for the intelligence brief
                    st.markdown("### Alfred's Report")
                    st.chat_message("assistant", avatar="").write(answer)
                    
                    # Add an "Export Report" button
                    st.download_button("Export Intelligence Brief", answer, file_name="alfred_report.txt")
                    
                else:
                    # Specific error parsing for the 500/401 issues we discussed
                    error_detail = response.json().get("detail", response.text)
                    st.error(f"Operational Interruption: {error_detail}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Connection Refused: Please ensure the FastAPI backend (main.py) is running on port 8000.")
            except Exception as e:
                st.error(f"Unexpected Anomaly: {str(e)}")

# FOOTER
st.markdown("---")
st.caption("Wayne Enterprises Confidential - Internal Use Only")