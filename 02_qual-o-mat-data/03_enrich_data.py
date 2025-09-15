import pandas as pd
import re
from tqdm import tqdm
import ollama

df = pd.read_json("wahlomat.json")

id_statement = df['id_statement']
text_statement = df['text_statement']

df_ref = pd.DataFrame({
    'id_statement':id_statement,
    'text_statement':text_statement,
})

df_ref = df_ref.drop_duplicates(ignore_index=True)

id_statement = df_ref['id_statement']
text_statement = df_ref['text_statement']

def get_paraphrases(sentence):
    prompt_kind = "Schreibe folgenden Satz so um, dass ein Kind ihn verstehen kann. Schreibe die Sätze so, als würden sie in der Zeitung stehen, ohne Überschrift. Erzeuge 10 Versionen mit jeweils 1 bis 5 Sätzen. Schreibe statt **Version 1** 1., statt **Version 2** 2. etc: "+sentence
    prompt_jugendlicher = "Schreibe folgenden Satz so um, dass ein Jugendlicher ihn verstehen kann. Schreibe die Sätze so, als würden sie in der Zeitung stehen, ohne Überschrift. Erzeuge 10 Versionen mit jeweils 1 bis 5 Sätzen. Schreibe statt **Version 1** 1., statt **Version 2** 2. etc: "+sentence
    prompt_erwachsener = "Schreibe folgenden Satz so um, dass ein Erwachsener ihn verstehen kann. Schreibe die Sätze so, als würden sie in der Zeitung stehen, ohne Überschrift. Erzeuge 10 Versionen mit jeweils 1 bis 5 Sätzen. Schreibe statt **Version 1** 1., statt **Version 2** 2. etc: "+sentence
    prompt_eloquenter_mensch = "Schreibe folgenden Satz so um, dass ein sehr eloquenter Mensch, der sich durch eine gehobene Wortwahl abzugrenzen versucht, ihn verstehen kann. Schreibe die Sätze so, als würden sie in der Zeitung stehen, ohne Überschrift. Erzeuge 10 Versionen mit jeweils 1 bis 5 Sätzen. Schreibe statt **Version 1** 1., statt **Version 2** 2. etc: "+sentence
    
    output_kind = ollama.chat(model='llama3.1:405b-instruct-q3_K_S', messages=[{'role': 'user', 'content': prompt_kind,},])
    output_jugendlicher = ollama.chat(model='llama3.1:405b-instruct-q3_K_S', messages=[{'role': 'user', 'content': prompt_jugendlicher,},])
    output_erwachsener = ollama.chat(model='llama3.1:405b-instruct-q3_K_S', messages=[{'role': 'user', 'content': prompt_erwachsener,},])
    output_eloquenter_mensch = ollama.chat(model='llama3.1:405b-instruct-q3_K_S', messages=[{'role': 'user', 'content': prompt_eloquenter_mensch,},])

    response_kind = output_kind['message']['content']
    response_jugendlicher = output_jugendlicher['message']['content']
    response_erwachsener = output_erwachsener['message']['content']
    response_eloquenter_mensch = output_eloquenter_mensch['message']['content']

    sentences_kind = [re.sub("[0-9]+. ", "", x) for x in response_kind.split("\n\n") if re.search("^[0-9]+. ", x)]
    sentences_jugendlicher = [re.sub("[0-9]+. ", "", x) for x in response_jugendlicher.split("\n\n") if re.search("^[0-9]+. ", x)]
    sentences_erwachsener = [re.sub("[0-9]+. ", "", x) for x in response_erwachsener.split("\n\n") if re.search("^[0-9]+. ", x)]
    sentences_eloquenter_mensch = [re.sub("[0-9]+. ", "", x) for x in response_eloquenter_mensch.split("\n\n") if re.search("^[0-9]+. ", x)]
    
    return sentences_kind, sentences_jugendlicher, sentences_erwachsener, sentences_eloquenter_mensch
    #return sentences_jugendlicher, sentences_erwachsener, sentences_eloquenter_mensch

l_id_statement = []
l_text_statement = []
l_sentences_kind = []
l_sentences_jugendlicher = []
l_sentences_erwachsener = []
l_sentences_eloquenter_mensch = []
input_ = [[x,y] for x,y in zip(id_statement, text_statement)]
#for id, text in tqdm(zip(id_statement, text_statement)):
for id, text in tqdm(input_):
    l_id_statement.append(id)
    l_text_statement.append(text)
    #sentences_jugendlicher, sentences_erwachsener, sentences_eloquenter_mensch = get_paraphrases(text)
    sentences_kind, sentences_jugendlicher, sentences_erwachsener, sentences_eloquenter_mensch = get_paraphrases(text)
    l_sentences_kind.append(sentences_kind)
    l_sentences_jugendlicher.append(sentences_jugendlicher)
    l_sentences_erwachsener.append(sentences_erwachsener)
    l_sentences_eloquenter_mensch.append(sentences_eloquenter_mensch)

    df_out = pd.DataFrame({
        'id_statement': l_id_statement,
        'text_statement': l_text_statement,
        'sentences_kind': l_sentences_kind,
        'sentences_jugendlicher': l_sentences_jugendlicher,
        'sentences_erwachsener': l_sentences_erwachsener,
        'sentences_eloquenter_mensch': l_sentences_eloquenter_mensch,
    })
    df_out.to_json("wahlomat_paraphrased_raw.json")