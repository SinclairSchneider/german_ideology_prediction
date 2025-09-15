import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from auto_gptq import exllama_set_max_input_length
from datasets import load_dataset
from tqdm import tqdm
import sys
import os

def main():
    if len(sys.argv) != 3:
        print("wrong number of command line arguments")
        return

    i_index = int(sys.argv[1])
    i_total = int(sys.argv[2])
    i_gpu = int(i_index%4)

    max_memory={0: "48GB", 1: "0GB", 2: "0GB", 3: "0GB"}
    if i_gpu == 0:
        max_memory={0: "48GB", 1: "0GB", 2: "0GB", 3: "0GB"}
    elif i_gpu == 1:
        max_memory={0: "0GB", 1: "48GB", 2: "0GB", 3: "0GB"}
    elif i_gpu == 2:
        max_memory={0: "0GB", 1: "0GB", 2: "48GB", 3: "0GB"}
    elif i_gpu == 3:
        max_memory={0: "0GB", 1: "0GB", 2: "0GB", 3: "48GB"}

    model_id = "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
      model_id,
      torch_dtype=torch.float16,
      low_cpu_mem_usage=True,
      device_map="auto",
      max_memory=max_memory,
    )

    model = exllama_set_max_input_length(model, 4096)

    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    ds = load_dataset("SinclairSchneider/Bundestagsreden_senti_pos_neg")
    df = ds['train'].to_pandas()
    df['index'] = list(df.index)

    l_index_bool = [((x-i_index)%i_total)==0 for x in range(len(df))]

    df_part = df[l_index_bool]

    l_Text = list(df_part['Text'])

    def get_summaries(text):
        prompt_kind = "Fasse die folgende Rede in 1 bis 5 Sätzen so zusammen, dass ein Kind sie verstehen kann. Beginne nicht immer mit \"Es wird\" oder \"Der Redner\"!!! Schreibe die Sätze in der passiven Form. Erwähne nur den Inhalt, nicht wer was gesagt oder gefordert hat. Erwähne auch keine Parteien oder Personen. Konzentriere dich auf Forderungen, aber beginne nicht immer mit \"es wird gefordert\". Beginne die Zusammenfassung mit Zusammenfassung:\n"+text
        output_kind = llm_pipeline([{"role": "user", "content": prompt_kind}], max_new_tokens=512, temperature=0.9, )
        response_kind = output_kind[0]['generated_text'][1]['content']
        if "Zusammenfassung:" in response_kind:
            summary_kind = response_kind.replace("Zusammenfassung:", "").strip()
        else:
            summary_kind = ""
    
        prompt_jugendlicher = "Fasse die folgende Rede in 1 bis 5 Sätzen so zusammen, dass ein Jugendlicher sie verstehen kann. Beginne nicht immer mit \"Es wird\" oder \"Der Redner\"!!! Schreibe die Sätze in der passiven Form. Erwähne nur den Inhalt, nicht wer was gesagt oder gefordert hat. Erwähne auch keine Parteien oder Personen. Konzentriere dich auf Forderungen, aber beginne nicht immer mit \"es wird gefordert\". Beginne die Zusammenfassung mit Zusammenfassung:\n"+text
        output_jugendlicher = llm_pipeline([{"role": "user", "content": prompt_jugendlicher}], max_new_tokens=512, temperature=0.9, )
        response_jugendlicher = output_jugendlicher[0]['generated_text'][1]['content']
        if "Zusammenfassung:" in response_jugendlicher:
            summary_jugendlicher = response_jugendlicher.replace("Zusammenfassung:", "").strip()
        else:
            summary_jugendlicher = ""
    
        prompt_erwachsener = "Fasse die folgende Rede in 1 bis 5 Sätzen so zusammen, dass ein Erwachsener sie verstehen kann. Beginne nicht immer mit \"Es wird\" oder \"Der Redner\"!!! Schreibe die Sätze in der passiven Form. Erwähne nur den Inhalt, nicht wer was gesagt oder gefordert hat. Erwähne auch keine Parteien oder Personen. Konzentriere dich auf Forderungen, aber beginne nicht immer mit \"es wird gefordert\". Beginne die Zusammenfassung mit Zusammenfassung:\n"+text
        output_erwachsener = llm_pipeline([{"role": "user", "content": prompt_erwachsener}], max_new_tokens=512, temperature=0.9, )
        response_erwachsener = output_erwachsener[0]['generated_text'][1]['content']
        if "Zusammenfassung:" in response_erwachsener:
            summary_erwachsener = response_erwachsener.replace("Zusammenfassung:", "").strip()
        else:
            summary_erwachsener = ""
    
        prompt_eloquenter_mensch = "Fasse die folgende Rede in 1 bis 5 Sätzen so zusammen, dass ein sehr eloquenter Mensch, der sich durch eine gehobene Wortwahl abzugrenzen versucht, sie verstehen kann. Beginne nicht immer mit \"Es wird\" oder \"Der Redner\"!!! Schreibe die Sätze in der passiven Form. Erwähne nur den Inhalt, nicht wer was gesagt oder gefordert hat. Erwähne auch keine Parteien oder Personen. Konzentriere dich auf Forderungen, aber beginne nicht immer mit \"es wird gefordert\". Beginne die Zusammenfassung mit Zusammenfassung:\n"+text
        output_eloquenter_mensch = llm_pipeline([{"role": "user", "content": prompt_eloquenter_mensch}], max_new_tokens=512, temperature=0.9, )
        response_eloquenter_mensch = output_eloquenter_mensch[0]['generated_text'][1]['content']
        if "Zusammenfassung:" in response_eloquenter_mensch:
            summary_eloquenter_mensch = response_eloquenter_mensch.replace("Zusammenfassung:", "").strip()
        else:
            summary_eloquenter_mensch = ""
    
        return summary_kind, summary_jugendlicher, summary_erwachsener, summary_eloquenter_mensch

    
    i = 0
    l_summary_kind = []
    l_summary_jugendlicher = []
    l_summary_erwachsener = []
    l_summary_eloquenter_mensch = []
    for text in tqdm(l_Text):
        summary_kind, summary_jugendlicher, summary_erwachsener, summary_eloquenter_mensch = get_summaries(text)
        
        l_summary_kind.append(summary_kind)
        l_summary_jugendlicher.append(summary_jugendlicher)
        l_summary_erwachsener.append(summary_erwachsener)
        l_summary_eloquenter_mensch.append(summary_eloquenter_mensch)
        
        df_temp = df_part.head(len(l_summary_kind)).copy()
        df_temp['summary_kind'] = l_summary_kind
        df_temp['summary_jugendlicher'] = l_summary_jugendlicher
        df_temp['summary_erwachsener'] = l_summary_erwachsener
        df_temp['summary_eloquenter_mensch'] = l_summary_eloquenter_mensch
        if i%10 == 0:
            df_temp.to_json("Bundestagsreden_senti_pos_neg_summarized_"+str(i_index+1)+"_"+str(i_total)+".json")
        if os.path.exists("kill_all.txt"):
            break
        i = i + 1

if __name__ == "__main__":
    main()