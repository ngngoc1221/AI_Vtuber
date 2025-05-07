import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import string
import time
from collections import Counter

import Google.gemini.GoogleAI_Gemini_API as gemini_api
import OpenAI.gpt.OpenAI_GPT_API as gpt_api










gemini_model_names_list = [name for name in gemini_api.gemini_models_max_input_tokens.keys()]
gpt_model_names_list = [name for name in gpt_api.gpt_models_max_input_tokens.keys()]
model_names_list = gemini_model_names_list + gpt_model_names_list





def Sentiment_Analysis_NLP(
        text,
        Emo_state_categories=["normal", "happy", "shy", "proud", "shock", "sad", "angry", "embarrass", "afraid", "confuse"],
        model="gemini-2.0-flash",
        timeout=5,
    ):
    global gemini_model_names_list, gpt_model_names_list

    SA_system_prompt = f"Bạn sẽ có nhiệm vụ phân tích văn bản được gửi đến và gửi lại các danh mục cảm xúc tương ứng bằng tiếng Anh. Theo các danh mục cảm xúc sau đây mà tôi đã định nghĩa: {Emo_state_categories}, chỉ cần gửi lại một cảm xúc có khả năng cao nhất, và chỉ trả về tên của cảm xúc đó. Đừng gửi lại cho tôi văn bản đã được gửi đến bạn."
    conversation = [
            {"role": "system", "content": SA_system_prompt},
            {"role": "user", "content": "(・_・;) Không, HuangCao, tại sao cậu lại đột nhiên nói vậy? Cậu đang cố ăn cắp con cá nhỏ của tôi à? (＠_＠;)＼(◎o◎)／!"},
            {"role": "assistant", "content": "confuse"},
            {"role": "user", "content": "HoangCao, cậu thật tệ (／‵Д′)／~ ╧╧ C0 ghét rau mùi nhất (╬ﾟдﾟ) Rau mùi và cần tây không nên tồn tại trên thế giới này (╯°□°)╯︵ ┻━┻"},
            {"role": "assistant", "content": "angry"},
            {"role": "user", "content": "(゜o゜) Ê ê ê! HoangCao, mày đang làm gì vậy? Đang học tiếng mèo kêu à? Cười chết mất! Thằng HoangCao này thật là kém cỏi! Haha! (≧∇≦)/ "}, # 🥲
            {"role": "assistant", "content": "happy"},
            {"role": "user", "content": text},
        ]

    start_time = time.time()

    if model in gemini_model_names_list:
        llm_result = gemini_api.run_with_timeout_GoogleAI_Gemini_API(
                conversation,
                "",
                model_name=model,
                max_output_tokens=10,
                temperature=0.2,
                timeout=timeout,
                retry=1,
                command="no_print",
            )

    elif model in gpt_model_names_list:
        llm_result = gpt_api.run_with_timeout_OpenAI_GPT_API(
                conversation,
                "",
                model_name=model,
                max_output_tokens=10,
                temperature=0.2,
                timeout=timeout,
                retry=1,
                command="no_print",
            )

    try:
        mcsw = most_common_specific_word(llm_result, Emo_state_categories)
    except Exception as e:
        print(f"\n{e}\n")
        mcsw = "normal"

    end_time = time.time()
    print("\nSentiment Analysis NLP ----------\n")
    print(f"Model: {model}")
    print(f"Duration: {end_time - start_time:.2f}s\n")
    print(f"Emotion State: {mcsw}")
    print("\n----------\n")
    return mcsw



def most_common_specific_word(text, Emo_state_categories):
    punctuation = string.punctuation + "："
    translator = str.maketrans(punctuation, " " * len(punctuation))

    text = text.lower()
    text = text.translate(translator)
    words = text.split()
    word_counts = Counter(words)

    Emo_state_categories = [word.lower() for word in Emo_state_categories]
    result = {word: word_counts[word] for word in Emo_state_categories}
    most_common_word = max(result, key=result.get)
    return most_common_word










if __name__ == "__main__":

    emo_state = Sentiment_Analysis_NLP(
            "Never gonna make you cry never gonna say goodbye"
        )




