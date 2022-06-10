import pandas as pd
from keybert import KeyBERT


def read_file():
    temp_df = pd.read_pickle("../datasets/clean_GLA.pkl")
    return temp_df


def main():
    temp_df = read_file()
    kw_model = KeyBERT()
    temp_sentence = " ".join(temp_df["words"][0])
    keywords = kw_model.extract_keywords(temp_sentence)
    print(keywords)


if __name__ == '__main__':
    main()
