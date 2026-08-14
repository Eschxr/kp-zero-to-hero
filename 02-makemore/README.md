# Makemore

Lesson #2: Intro to language modeling

Makemore takes lines of text and produces more "things" akin to what it's seen
Apparently it's an "autoregressive character-level language model" (whatever that means) which I'll have to figure out down the line

names.txt is a dataset of human names compiled by Andrej Karpathy, which'll be the main target for makemore

so "character-level" just means the model treats each "item" as sequences of individual characters
and eventually it learns to predict the next character in a sequence

now I just gotta figure out what "autoregressive" means

