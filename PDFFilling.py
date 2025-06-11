from pypdf import PdfReader, PdfWriter
import json
import pickle
import os

class PDFFormFiller:
    def __init__(self, mappings_path="filling_input/mappings/", template_path="filling_input/i-130.pdf", output_path="output/filled_form.pdf"):
        self.mappings_path = mappings_path
        self.template_path = template_path
        self.output_path = output_path
        self.mappings = self._load_mappings()


    def _load_mappings(self):
        mappings = {}
        for filename in os.listdir(self.mappings_path):
            if filename.endswith(".pkl"):
                with open(os.path.join(self.mappings_path, filename), "rb") as file:
                    data = pickle.load(file)
                    if isinstance(data, dict):
                        mappings.update(data)
        return mappings

    def _flatten_json(self, nested_dict, parent_key='', sep='.'):
        items = {}
        for key, value in nested_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                items.update(self._flatten_json(value, new_key, sep))
            elif value != "NA":
                items[new_key] = value
        return items

    def fill_checkbox(self, writer, idx, field_name, field_value, value):
        # [same fill_checkbox code here — unchanged for brevity]
        pass

    def fill(self, json_path_or_data, field_names=False):
        if not field_names and isinstance(json_path_or_data, str) and json_path_or_data.endswith('.json'):
            with open(json_path_or_data, "r", encoding="utf-8") as json_file:
                raw_data = json.load(json_file)
        elif isinstance(json_path_or_data, str):
            raw_data = json.loads(json_path_or_data)
        else:
            raw_data = json_path_or_data

        json_data = self._flatten_json(raw_data)

        reader = PdfReader(self.template_path)
        writer = PdfWriter()
        writer.append(reader)

        for idx, page in enumerate(reader.pages):
            if "/Annots" in page:
                for annot in page["/Annots"]:
                    annot_obj = annot.get_object()
                    field_name = annot_obj.get("/T", "Unnamed Field")

                    if not field_names and field_name in self.mappings:
                        json_key = self.mappings[field_name]
                        if json_key in json_data and json_data[json_key]:
                            # Handle checkboxes, text, and dropdowns
                            value = json_data[json_key]
                            if annot_obj.get("/FT") == "/Tx":
                                writer.update_page_form_field_values(writer.pages[idx], {field_name: value}, auto_regenerate=False)
                            elif annot_obj.get("/FT") == "/Btn":
                                if "/AP" in annot_obj and "/N" in annot_obj["/AP"]:
                                    values = list(annot_obj["/AP"]["/N"].keys())
                                    cb_value = next((v for v in values if v != "/Off"), None)
                                    self.fill_checkbox(writer, idx, field_name, value, cb_value)
                            elif annot_obj.get("/FT") == "/Ch":
                                writer.update_page_form_field_values(writer.pages[idx], {field_name: value}, auto_regenerate=False)

                    elif field_names and field_name in json_data:
                        writer.update_page_form_field_values(writer.pages[idx], {field_name: json_data[field_name]})

        if os.path.exists(self.output_path):
            os.remove(self.output_path)

        with open(self.output_path, "wb") as f_out:
            writer.write(f_out)

        return self.output_path

    def get_missing_fields(self):
        reader = PdfReader(self.template_path)
        textfields, check_boxes, dropdowns = [], [], []
        for page in reader.pages:
            if "/Annots" in page:
                for annot in page["/Annots"]:
                    annot_obj = annot.get_object()
                    field_name = annot_obj.get("/T", "Unnamed Field")
                    field_value = annot_obj.get("/V", "")
                    if annot_obj.get("/FT") == "/Tx" and field_value in [None, ""]:
                        textfields.append(field_name)
                    elif annot_obj.get("/FT") == "/Btn" and field_value in ["/Off", "/off", ""]:
                        check_boxes.append(field_name)
                    elif annot_obj.get("/FT") == "/Ch" and field_value == "":
                        dropdowns.append(field_name)
        return textfields, check_boxes, dropdowns



