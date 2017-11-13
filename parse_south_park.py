data_path = 'conv/All-seasons.csv'
parsed_file = 'conv/south_park.txt';
num_samples = 10000
max_length = 64
import csv
import re

lines = open(data_path).read().split('\n')
sentence_endings = r"[.]"
F = open(parsed_file, "w")
prevline = ''
i = 0;
total_length = 0;
for line in csv.reader(lines, quotechar='"', delimiter=',',
                       quoting=csv.QUOTE_ALL, skipinitialspace=True):
    try:
        if i > num_samples:
            break

        i = i + 1;

        input_text = prevline
        target_text = line[3].lower()
        total_length = total_length + len(target_text);
        if len(target_text) > max_length:
            target_text = target_text[0:max_length];
            target_text = target_text[0: target_text.rfind(' ')];

        #min_length = min(target_text.lfind('.'),target_text.lfind('!'),target_text.lfind('?'))
        target_text = re.split(sentence_endings,target_text)

        target_text=target_text[0]

        prevline = target_text;
        if i < 3:
            continue
        print input_text + '\t' + target_text + '\n'
        if "i m " not in target_text:
            F.write(input_text + '\t' + target_text + '\n');
    except:
        print "This is an error message!"

F.close()

print ('average length =', total_length / (i - 1));
