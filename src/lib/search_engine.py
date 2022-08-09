# import spacy
# from rank_bm25 import BM25Okapi
# from tqdm import tqdm
# import pandas as pd

# # TODO: Search engine for ocr descriptions

# dir = '/Users/alexdrozdz/Desktop/Studia/00. Seminarium magisterskie/Master_degree/ocr_results/vc_ocr_data.json'

# ocr_data = json.load(open(dir))

# # print(ocr_data['vc_10593.png'])

# df = pd.DataFrame()
# df['file'] = ['vc_10593.png', 'vc_10592.png', 'vc_10591.png']
# df['text'] = [
#     f"{ocr_data['vc_10593.png']['predicted_label']} {ocr_data['vc_10593.png']['cleaned_text']}",
#     f"{ocr_data['vc_10592.png']['predicted_label']} {ocr_data['vc_10592.png']['cleaned_text']}",
#     f"{ocr_data['vc_10591.png']['predicted_label']} {ocr_data['vc_10591.png']['cleaned_text']}",
# ]

# print(df)

# nlp = spacy.load("en_core_web_sm")
# text_list = df.text.str.lower().values
# tok_text=[] # for our tokenised corpus
# #Tokenising using SpaCy:
# for doc in tqdm(nlp.pipe(text_list, disable=["tagger", "parser","ner"])):
#    tok = [t.text for t in doc if t.is_alpha]
#    tok_text.append(tok)


