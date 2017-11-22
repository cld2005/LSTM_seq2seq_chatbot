import string
data_path_encode = 'conv/train.enc'
data_path_decode = 'conv/train.dec'
parsed_file = 'conv/codedak_conv.txt';
num_samples = 10000
encode_lines = open(data_path_encode).read().split('\n')
decode_lines = open(data_path_decode).read().split('\n')


print ('encode_lines len ', len(encode_lines))
print ('decode_lines len ', len(decode_lines))

lines = zip(encode_lines,decode_lines)
print ('decode_lines lines ', len(lines))

encode_set =set()
decode_set = set()
F = open(parsed_file,"w")

for line in lines[: min(num_samples, len(lines) - 1) :]:
    input_text = string.lower(line[0].translate(None, string.punctuation))
    target_text = string.lower(line[1].translate(None, string.punctuation))

    for txt in input_text.split(' '):
        if txt not in encode_set:
            encode_set.add(txt)

    for txt in target_text.split(' '):
        if txt not in decode_set:
            decode_set.add(txt)
    print input_text+'\t'+target_text+'\n'
    F.write(input_text+'\t'+target_text+'\n');
F.close()

print ('encode_set size ', len(encode_set))
print ('decode_set size ', len(decode_set))

