This is a Python package called "lm_kit" which is made to be installed in Google Colab for training small language models. It is made to be user friendly and powerful, allowing almost anyone to tweak language models. That is the key distinction.
It is built with the assumption that the user connects a high-end GPU like an A100 or a T4.

The access point of the package is called "kit". Here is an example of how it could be used once installed inside of Google Colab:

`from lm_kit import kit

dataset = kit.get_dataset("hugging_face", "roneneldan/TinyStories", "train[:5%]")
model = kit.create_model(dataset, model_size = "30M", steps=15000)
model = model.train()
model.complete("Once upon a time") # All ready to go. Simple as that.
model.save(".") # Save it, and deploy anywhere.
`