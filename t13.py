from textgenrnn import textgenrnn

textgen = textgenrnn()
textgen.train_from_file('hacker_news_2000.txt', num_epochs=1)
textgen.generate()