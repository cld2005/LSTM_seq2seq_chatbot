data_path = 'conv/pairs.txt'
parsed_file = 'conv/parsed.txt';
num_samples = 10000
lines = open(data_path).read().split('\n')

F = open(parsed_file,"w")

for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split(',')
    print input_text[1:-1]+'\t'+target_text[1:-2]+'\n'
    if "i m " not in target_text:
     F.write(input_text[1:-1]+'\t'+target_text[1:-2]+'\n');
F.close()


