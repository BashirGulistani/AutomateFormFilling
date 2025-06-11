import json
import re
import os
import logging
import gradio as gr
from openai import OpenAI
from dataExtraction import DocumentProcessor

logging.basicConfig(level=logging.INFO)

class GradioChatBot:
    sys_prompt = (
        "You are an assistant helping fill in missing fields in a form. "
        "You will either ask a short friendly question for a field, explain what a field means if asked, "
        "or clean up a user input and return just the value with no comments."
    )

    def __init__(self, path, model="gpt-4", mode="petitioner"):
        with open(path, "r") as f:
            self.data = json.load(f)

        self.client = OpenAI(api_key="")
        self.model = model
        self.mode = mode
        self.fields = self._get_missing_fields()
        self.current_field = None
        self.chat_history = [{"role": "assistant", "content": ""}]
        self.changed = False

    def _get_missing_fields(self):
        def walk(d, prefix=[]):
            result = []
            for k, v in d.items():
                if isinstance(v, dict):
                    result.extend(walk(v, prefix + [k]))
                elif v == "NA":
                    result.append(prefix + [k])
            return result
        return walk(self.data)

    def _query_model(self, user_content):
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": user_content}
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI query failed: {e}")
            return ""

    def _ask_question(self, path):
        label = " ".join(path).replace("_", " ")
        return self._query_model(f"Ask one friendly and short question to collect this info: {label}. No explanation.")

    def _is_user_asking(self, user_input):
        return "?" in user_input or user_input.lower().startswith(("what", "where", "how", "do i"))

    def _explain_field(self, path, user_input):
        label = " ".join(path).replace("_", " ")
        return self._query_model(f"The user asked: '{user_input}'. Explain politely what '{label}' means and how to find it.")

    def _extract_value(self, path, user_input):
        label = path[-1].replace("_", " ")
        value = self._query_model(f"The user said: '{user_input}' for the field '{label}'. Return just the cleaned value. No comments. Don't add anything extra on numbers like adding hyphens please.")
        return re.sub(r"\s*\([^)]*\)$", "", value).strip()

    def _update_field(self, path, value):
        ref = self.data
        for key in path[:-1]:
            ref = ref[key]
        ref[path[-1]] = value
        self.fields = self._get_missing_fields()
        self.changed = True

    def chat(self, user_input):
        if self.current_field is None:
            if not self.fields:
                return self._complete_stage()
            self.current_field = self.fields[0]
            question = self._ask_question(self.current_field)
            self.chat_history.append({"role": "assistant", "content": question})
            return self.chat_history, "Awaiting your response.", None

        if user_input.lower() == "skip":
            self._update_field(self.current_field, "")
            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": "Okay, skipping this one."})
            self.current_field = None

            if not self.fields:
                return self._complete_stage()

            self.current_field = self.fields[0]
            next_question = self._ask_question(self.current_field)
            self.chat_history.append({"role": "assistant", "content": next_question})
            return self.chat_history, "Next question coming...", None

        self.chat_history.append({"role": "user", "content": user_input})

        if self._is_user_asking(user_input):
            explanation = self._explain_field(self.current_field, user_input)
            self.chat_history.append({"role": "assistant", "content": explanation})
            return self.chat_history, "Let me know if you need more info.", None

        value = self._extract_value(self.current_field, user_input)
        if value.lower() in ["", "na", "none", "not sure"]:
            self.chat_history.append({"role": "assistant", "content": "I couldn't understand that. Could you rephrase?"})
            return self.chat_history, "Unclear input.", None

        self._update_field(self.current_field, value)
        self.current_field = None

        if not self.fields:
            return self._complete_stage()

        self.current_field = self.fields[0]
        next_question = self._ask_question(self.current_field)
        self.chat_history.append({"role": "assistant", "content": next_question})
        return self.chat_history, "Next question coming...", None

    def _complete_stage(self):
        self._save("FinalPetitioner.json")
        # Generate PDF
        from PDFFilling import PDFFormFiller
        filler = PDFFormFiller()
        pdf_path = filler.fill("FinalPetitioner.json")

        return self.chat_history, "Form complete.", pdf_path

    def _save(self, out):
        if self.changed:
            with open(out, "w") as f:
                json.dump(self.data, f, indent=2)

def respond(message, history):
    chat, status, pdf_path = bot.chat(message)
    if pdf_path:
        return chat, "", gr.update(value=pdf_path, visible=True)
    else:
        return chat, "", gr.update(visible=False)

def handle_upload(files):
    os.makedirs("petitioner_documents", exist_ok=True)
    for file in files:
        file_path = getattr(file, "name", None)
        if file_path and os.path.exists(file_path):
            dest = os.path.join("petitioner_documents", os.path.basename(file_path))
            with open(file_path, "rb") as f_in, open(dest, "wb") as f_out:
                f_out.write(f_in.read())
    processor = DocumentProcessor("petitioner_documents")
    processor.processDocs()
    extracted_json1 = processor.runPipeline(processor.petitionerPrompt1())

    with open("filling_input/pet1.json") as f1:
        template_json1 = json.load(f1)
    corrected_json1 = processor.correct_json(extracted_json1, template_json1)

    with open("FinalPetitioner.json", "w") as f:
        json.dump(corrected_json1, f, indent=2)

    global bot
    bot = GradioChatBot("FinalPetitioner.json", mode="petitioner")
    chat, _, _ = bot.chat("")
    return chat

def generate_filled_pdf(json_path="FinalPetitioner.json"):
    from PDFFilling import PDFFormFiller
    filler = PDFFormFiller(template_path="filling_input/i-130.pdf", output_path="output/filled_form.pdf")
    return filler.fill(json_path)

next_upload_handler = handle_upload

with gr.Blocks() as interface:
    gr.Markdown("# I-130 Form Assistant")
    chatbot = gr.Chatbot(
        value=[{"role": "assistant", "content": "Welcome! Please upload documents related to the petitioner, spouse, and parents. Rename each file clearly, like `petitioner_passport.pdf` or `parent1_ID.jpg`"}],
        type="messages",
        render_markdown=False
    )
    msg = gr.Textbox(placeholder="Type your answer or ask a question...")
    pdf_output = gr.File(label="Download Filled PDF", visible=False)
    msg.submit(respond, [msg, chatbot], [chatbot, msg, pdf_output])
    with gr.Row():
        file_upload = gr.File(label="Upload Documents", file_types=[".pdf", ".jpg", ".png"], file_count="multiple")
        upload_button = gr.Button("Extract Data")
        upload_button.click(
            lambda files: (next_upload_handler(files), gr.update(value=None)),
            inputs=[file_upload],
            outputs=[chatbot, file_upload]
        )

interface.launch(share=True)
