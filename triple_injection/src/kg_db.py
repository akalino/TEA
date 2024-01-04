from transformers import DistilBertConfig, DistilBertModel


if __name__ == "__main__":
    configuration = DistilBertConfig()
    model = DistilBertModel(configuration)
    print(model)
