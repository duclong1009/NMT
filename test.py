from vncorenlp import VnCoreNLP


vi_tokenizer = VnCoreNLP(
    "./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size="-Xmx500m"
)
text = "Xin chào tôi tên là Nguyễn Đức Long."
print(vi_tokenizer.tokenize(text))
from collections import Counter

counter = Counter()
counter.update(vi_tokenizer.tokenize(text)[0])
print(counter)
