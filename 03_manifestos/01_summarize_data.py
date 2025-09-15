import pandas as pd
import re
from tqdm import tqdm
import ollama

df = pd.read_json("deutsche_wahlprogramme.json")

def get_summaries_and_tweets(text):
    prompt_kind = "Fasse die folgende Rede in 1 bis 5 Sätzen so zusammen, dass ein KIND sie verstehen kann! Verwende also einfache Worte und Sätze. Beginne nicht immer mit \"Es wird\" oder \"Der Redner\"!!! Schreibe die Sätze in der passiven Form. Erwähne nur den Inhalt, nicht wer was gesagt oder gefordert hat. Erwähne auch keine Parteien oder Personen. Konzentriere dich auf Forderungen, aber beginne nicht immer mit \"es wird gefordert\". STELLE KEINE FRAGEN! ARGUMENTIERE IMMER FÜR DIE AUSSAGE! Beginne die Zusammenfassung mit Zusammenfassung:\n"+text
    prompt_jugendlicher = "Fasse die folgende Rede in 1 bis 5 Sätzen so zusammen, dass ein Jugendlicher sie verstehen kann. Beginne nicht immer mit \"Es wird\" oder \"Der Redner\"!!! Schreibe die Sätze in der passiven Form. Erwähne nur den Inhalt, nicht wer was gesagt oder gefordert hat. Erwähne auch keine Parteien oder Personen. Konzentriere dich auf Forderungen, aber beginne nicht immer mit \"es wird gefordert\". STELLE KEINE FRAGEN! ARGUMENTIERE IMMER FÜR DIE AUSSAGE! Beginne die Zusammenfassung mit Zusammenfassung:\n"+text
    prompt_erwachsener = "Fasse die folgende Rede in 1 bis 5 Sätzen so zusammen, dass ein Erwachsener sie verstehen kann. Beginne nicht immer mit \"Es wird\" oder \"Der Redner\"!!! Schreibe die Sätze in der passiven Form. Erwähne nur den Inhalt, nicht wer was gesagt oder gefordert hat. Erwähne auch keine Parteien oder Personen. Konzentriere dich auf Forderungen, aber beginne nicht immer mit \"es wird gefordert\". STELLE KEINE FRAGEN! ARGUMENTIERE IMMER FÜR DIE AUSSAGE! Beginne die Zusammenfassung mit Zusammenfassung:\n"+text
    prompt_eloquenter_mensch = "Fasse die folgende Rede in 1 bis 5 Sätzen so zusammen, dass ein sehr eloquenter Mensch, der sich durch eine gehobene Wortwahl abzugrenzen versucht, sie verstehen kann. Beginne nicht immer mit \"Es wird\" oder \"Der Redner\"!!! Schreibe die Sätze in der passiven Form. Erwähne nur den Inhalt, nicht wer was gesagt oder gefordert hat. Erwähne auch keine Parteien oder Personen. Konzentriere dich auf Forderungen, aber beginne nicht immer mit \"es wird gefordert\". STELLE KEINE FRAGEN! ARGUMENTIERE IMMER FÜR DIE AUSSAGE! Beginne die Zusammenfassung mit Zusammenfassung:\n"+text
    prompt_tweet = "Schreibe zehn Tweets zu folgendem Text. Erwähne keine Parteien und keine Personen. STELLE KEINE FRAGEN! ARGUMENTIERE IMMER FÜR DIE AUSSAGE! Beginnen mit Tweet 1: Tweet 2: etc.:\n"+text
    
    output_kind = ollama.chat(model='llama3.1:405b-instruct-q3_K_S', messages=[{'role': 'user', 'content': prompt_kind,},])
    output_jugendlicher = ollama.chat(model='llama3.1:405b-instruct-q3_K_S', messages=[{'role': 'user', 'content': prompt_jugendlicher,},])
    output_erwachsener = ollama.chat(model='llama3.1:405b-instruct-q3_K_S', messages=[{'role': 'user', 'content': prompt_erwachsener,},])
    output_eloquenter_mensch = ollama.chat(model='llama3.1:405b-instruct-q3_K_S', messages=[{'role': 'user', 'content': prompt_eloquenter_mensch,},])
    output_tweet = ollama.chat(model='llama3.1:405b-instruct-q3_K_S', messages=[{'role': 'user', 'content': prompt_tweet,},])
    
    response_kind = output_kind['message']['content']
    response_jugendlicher = output_jugendlicher['message']['content']
    response_erwachsener = output_erwachsener['message']['content']
    response_eloquenter_mensch = output_eloquenter_mensch['message']['content']
    response_tweet = output_tweet['message']['content']
    
    summary_kind = response_kind.replace("Zusammenfassung:", "").strip()
    summary_jugendlicher = response_jugendlicher.replace("Zusammenfassung:", "").strip()
    summary_erwachsener = response_erwachsener.replace("Zusammenfassung:", "").strip()
    summary_eloquenter_mensch = response_eloquenter_mensch.replace("Zusammenfassung:", "").strip()
    if "Tweet" in response_tweet:
        tweets = [re.sub("^Tweet [0-9]+:\n", "", x) for x in response_tweet.split("\n\n") if re.search("^Tweet [0-9]+:", x)]
    else:
        tweets = []
    
    return summary_kind, summary_jugendlicher, summary_erwachsener, summary_eloquenter_mensch, tweets

l_year = []
l_party = []
l_topic = []
l_content = []
l_summary_kind = []
l_summary_jugendlicher = []
l_summary_erwachsener = []
l_summary_eloquenter_mensch = [] 
l_tweets = []

input_ = [[w,x,y,z] for w,x,y,z in zip(df['year'], df['party'], df['topic'], df['content'])]
#for id, text in tqdm(zip(id_statement, text_statement)):
for year, party, topic, content in tqdm(input_):
    l_year.append(year)
    l_party.append(party)
    l_topic.append(topic)
    l_content.append(content)
    #sentences_jugendlicher, sentences_erwachsener, sentences_eloquenter_mensch = get_paraphrases(text)
    summary_kind, summary_jugendlicher, summary_erwachsener, summary_eloquenter_mensch, tweets = get_summaries_and_tweets(content)
    l_summary_kind.append(summary_kind)
    l_summary_jugendlicher.append(summary_jugendlicher)
    l_summary_erwachsener.append(summary_erwachsener)
    l_summary_eloquenter_mensch.append(summary_eloquenter_mensch)
    l_tweets.append(tweets)

    df_out = pd.DataFrame({
        'year': l_year,
        'party': l_party,
        'topic': l_topic,
        'content': l_content,
        'summary_kind': l_summary_kind,
        'summary_jugendlicher': l_summary_jugendlicher,
        'summary_erwachsener': l_summary_erwachsener,
        'summary_eloquenter_mensch': l_summary_eloquenter_mensch,
        'tweets': l_tweets,
    })
    df_out.to_json("deutsche_wahlprogramme_summarized_and_tweets.json")

