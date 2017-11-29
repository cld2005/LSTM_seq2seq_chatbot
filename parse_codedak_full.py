import string
data_path_encode = 'conv/train.enc'
data_path_decode = 'conv/train.dec'
parsed_file = 'conv/codedak_conv_full.txt';
num_samples = 100000
encode_lines = open(data_path_encode).read().split('\n')
decode_lines = open(data_path_decode).read().split('\n')


print ('encode_lines len ', len(encode_lines))
print ('decode_lines len ', len(decode_lines))

lines = zip(encode_lines,decode_lines)
print ('decode_lines lines ', len(lines))

encode_set = {}
decode_set = {}
F = open(parsed_file,"w")

for line in lines[: min(num_samples, len(lines) - 1) :]:
    input_text = string.lower(line[0].translate(None, [string.punctuation,'\t']))
    target_text = string.lower(line[1].translate(None, [string.punctuation,'\t']))

    for txt in input_text.split(' '):
        if not encode_set.has_key(txt):
            encode_set[txt]=0
        encode_set[txt]=encode_set[txt]+1

    for txt in target_text.split(' '):
        if not decode_set.has_key(txt):
            decode_set[txt] = 0
        decode_set[txt] = decode_set[txt] + 1
    #print input_text+'\t'+target_text+'\n'
    F.write(input_text+'\t'+target_text+'\n');
F.close()

print ('encode_set size ', len(encode_set))
print ('decode_set size ', len(decode_set))


common_encode_set = dict ((k,v) for (k,v) in encode_set.items() if v >30)
print ('common_encode_set size f>30 ', len(common_encode_set))
common_decode_set = dict ((k,v) for (k,v) in decode_set.items() if v >30)
print ('common_decode_set size f>30', len(common_decode_set))

