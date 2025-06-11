import re
import json
import logging
import pytesseract
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.builders import PromptBuilder
from pypdf import PdfReader
from pdf2image import convert_from_path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

import os


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DocumentProcessor:

    def __init__(self, folder_path):
        from openai import OpenAI
        self.client = OpenAI(api_key="")

        self.folder_path = folder_path
        self.document_store = InMemoryDocumentStore()

    def extractPDF(self, file_path):
        try:
            pdf = PdfReader(file_path)
            return "\n".join([page.extract_text() or "" for page in pdf.pages]).strip()
        except Exception as e:
            logging.warning(f"Error extracting text from {file_path}: {e}")
            return ""

    def extractImg(self, file_path):
        try:
            img = Image.open(file_path)
            return pytesseract.image_to_string(img).strip()
        except Exception as e:
            logging.error(f"Error reading image {file_path}: {e}")
            return ""

    def extractPDFIMG(self, file_path):
        try:
            images = convert_from_path(file_path, dpi=300)
            return "\n".join([pytesseract.image_to_string(img).strip() for img in images]).strip()
        except Exception as e:
            logging.error(f"OCR error on {file_path}: {e}")
            return ""

    def processFiles(self, file_name):
        file_path = os.path.join(self.folder_path, file_name)
        text = ""

        if file_name.lower().endswith(".pdf"):
            text = self.extractPDF(file_path)
            if not text:
                logging.info(f"No text found in {file_name}, applying OCR...")
                text = self.extractPDFIMG(file_path)

        elif file_name.lower().endswith((".jpg", ".png")):
            logging.info(f"Applying OCR on image: {file_name}")
            text = self.extractImg(file_path)

        return Document(content=text, meta={"file": file_name}) if text else None

    def processDocs(self):
        files = [f for f in os.listdir(self.folder_path) if f.endswith((".pdf", ".jpg", ".png"))]

        with ThreadPoolExecutor() as executor:
            docs = list(filter(None, executor.map(self.processFiles, files)))

        if not docs:
            logging.warning("No valid documents found. Extraction will fail.")
            return False

        for doc in docs:
            logging.info(f"Extracted from {doc.meta['file']}:\n{doc.content[:500]}")
        self.document_store.write_documents(docs)
        return True

    def runPipeline(self, prompt):
        pipeline = Pipeline()
        retriever = InMemoryBM25Retriever(document_store=self.document_store)
        prompt_builder = PromptBuilder(template=prompt)


        llm_generator = OpenAIGenerator(
            api_key=Secret.from_token(""),
            model="gpt-4o-mini"
        )

        pipeline.add_component("retriever", retriever)
        pipeline.add_component("prompt_builder", prompt_builder)
        pipeline.add_component("llm", llm_generator)

        pipeline.connect("retriever", "prompt_builder.documents")
        pipeline.connect("prompt_builder", "llm")

        response = pipeline.run({"retriever": {"query": "Extract the required details"}})

        llm_output = response.get("llm", {}).get("replies", "No response received.")

        if isinstance(llm_output, list) and llm_output:
            llm_output = llm_output[0]

        cleaned_output = self.cleanLLMOutput(llm_output)

        try:
            return json.loads(cleaned_output)
        except json.JSONDecodeError as e:
            logging.error(f"JSON Parsing Error. Raw response: {llm_output}")
            return {}


    def cleanLLMOutput(self, llm_output):
        # Remove comments
        json_start = llm_output.find("{")
        json_end = llm_output.rfind("}") + 1
        llm_output = llm_output[json_start:json_end]
        # Quotes
        llm_output = re.sub(r"([a-zA-Z0-9_]+):", r'"\1":', llm_output)
        llm_output = re.sub(r": ([A-Za-z0-9_]+)(,|})", r': "\1"\2', llm_output)

        return llm_output

    def correct_json(self, json_output: dict, json_template: dict):
        json_corrector_prompt = """
    You are a strict JSON corrector.

    You will receive two inputs:

    1. `json_data`: a raw or malformed JSON object  
    2. `template_data`: a JSON object that represents the desired structure

    Your task is to:
    - Transform the `json_data` to match the exact field structure of `template_data`
    - Do not add new fields or change field names
    - If a field from the template is missing in `json_data`, fill it with "NA"
    - If a value exists but is empty or irrelevant, also fill it with "NA"
    - Use the format {{FIELD_NAME}} as placeholder.

    Output only a single valid JSON object — nothing else. No comments. No explanations. No multiple JSONs.

    Here is the input:

    JSON:
    {json_data}

    TEMPLATE:
    {template_data}

    OUTPUT:
        """

        prompt = json_corrector_prompt.format(
            json_data=json.dumps(json_output, indent=2),
            template_data=json.dumps(json_template, indent=2)
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            reply = response.choices[0].message.content.strip()
            return json.loads(reply)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse corrected JSON: {e}\nRaw response:\n{reply}")
            return {}
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            return {}


    def petitionerPrompt1(self):
        return """
        These are extracted documents:
        {% for document in documents %}
            DOCUMENT {{ document.meta.file }}:
            {{ document.content }}

            END DOCUMENT {{ document.meta.file }}.
        {% endfor %}

        Don't add any COMMENTS. Extract the following petitioner details if available; otherwise, return "NA":        
        - petitioner_DocumentNumber/AlienRegistrationNumber
        - petitioner_USCIS_Online_Account_Number
        - petitioner_Social_Security_Number
        - petitioner_Last_Name
        - petitioner_First_Name
        - petitioner_Middle_Name
        - petitioner_CityofBirth
        - petitioner_CountryofBirth
        - petitioner_DoB (mm/dd/yyyy)
        - petitioner_Sex
        
        Format output in JSON.
        """





