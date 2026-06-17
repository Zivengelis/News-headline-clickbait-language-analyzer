import re
import csv

def normalize(text):
    text = text.strip()

    # Foto: ..., Video: ..., Intervija: ...
    text = re.sub(r'^\S+:\s*', '', text)

    # FOTO ⟩ ..., REDZI TĀLĀK! ⟩ ...
    text = re.sub(r'^.*?⟩\s*', '', text)

    # noņem komentāru skaitu beigās
    text = re.sub(r'\s*\(\d+\)\s*$', '', text)

    # noņem liekās atstarpes
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


unique_headlines = set()

with open("headlines.txt", "r", encoding="utf-8") as f:
    for line in f:
        headline = normalize(line)

        if headline:
            unique_headlines.add(headline)

with open("headlines_for_labeling_2.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow(["label", "headline"])

    for headline in sorted(unique_headlines):
        writer.writerow(["", headline])

print(f"Saglabāti {len(unique_headlines)} unikāli virsraksti.")