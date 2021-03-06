C_KEYWORDS = set(["auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long", "register", "restrict", "return", "short", "signed", "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned", "void", "volatile", "while", "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex", "_Generic", "_Imaginary", "_Noreturn", "_Static_assert", "_Thread_local", "__func__"])


def get_c_identifier_mapping(code):
    identifier_mapping = []

    in_string = False
    for token in code:
        if not in_string:
            if token in C_KEYWORDS:
                identifier_mapping.append(0)
            elif token[0] == '"':
                in_string = True
                identifier_mapping.append(0)
                if token[-1] == '"':
                    if len(token) > 1:
                        if token[-2] != '\\':
                            in_string = False
                    else:
                        in_string = False
            elif token[0].isalpha() or token[0] == '_': # should I add '$'? 
                identifier_mapping.append(1)
            else:
                identifier_mapping.append(0)
        elif token[-1] == '"':
            in_string = False
            identifier_mapping.append( 0 )
        else:
            identifier_mapping.append( 0 )
    
    return identifier_mapping


def token_to_subtoken_map(id_map, subtokenized_code, code=None):
    subtoken_id_map = []

    token_index = 0
    i = 0
    while i < len(subtokenized_code):
        subtoken_id_map.append(id_map[token_index])
        # print('put', id_map[token_index])
        if not subtokenized_code[i].endswith('@@'):
            # print('token_index', token_index, len(subtokenized_code) - i, code[token_index], subtokenized_code[i])
            token_index += 1
        i += 1
    return subtoken_id_map


if __name__ == "__main__":
    file = "./data/c/test"
    
    bpe2000_file = "./data/c/merge/test"
    bpe5000_file = "./data/c/ori/test"
    bpe10000_file = "./data/c/split/test"

    
    bpe2000_map_file = "./data/c/merge/id_test"
    bpe5000_map_file = "./data/c/ori/id_test"
    bpe10000_map_file = "./data/c/split/id_test"
    
    with open(file, 'r') as fr, open(bpe10000_file, 'r') as f_bpe10000, open(bpe10000_map_file, 'w') as fw_bpe10000,\
        open(bpe2000_file, 'r') as f_bpe2000,\
        open(bpe5000_file, 'r') as f_bpe5000,\
        open(bpe2000_map_file, 'w') as fw_bpe2000, \
        open(bpe5000_map_file, 'w') as fw_bpe5000:
        
        for line, bpe2000_line, bpe5000_line, bpe10000_line in zip(fr, f_bpe2000, f_bpe5000, f_bpe10000):
        #for line, bpe10000_line in zip(fr, f_bpe10000):
            # use in c
            # code = line.rstrip('\n')[4: -5].split()
            # code_bpe2000 = bpe2000_line.rstrip('\n')[4: -5].split()
            # code_bpe5000 = bpe5000_line.rstrip('\n')[4: -5].split()
            # code_bpe10000 = bpe10000_line.rstrip('\n')[4: -5].split()


            code = line.rstrip('\n').split()
            code_bpe2000 = bpe2000_line.rstrip('\n').split()
            code_bpe5000 = bpe5000_line.rstrip('\n').split()
            code_bpe10000 = bpe10000_line.rstrip('\n').split()
            
            identifier_mapping = get_c_identifier_mapping(code)
            assert(len(identifier_mapping) == len(code))
            bpe2000_id_mapping = token_to_subtoken_map(identifier_mapping, code_bpe2000)
            bpe5000_id_mapping = token_to_subtoken_map(identifier_mapping, code_bpe5000)
            bpe10000_id_mapping = token_to_subtoken_map(identifier_mapping, code_bpe10000)
            
            fw_bpe2000.write(str(bpe2000_id_mapping))
            fw_bpe2000.write('\n')
            
            fw_bpe5000.write(str(bpe5000_id_mapping))
            fw_bpe5000.write('\n')

            fw_bpe10000.write(str(bpe10000_id_mapping))
            fw_bpe10000.write('\n')
