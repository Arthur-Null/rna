from eden.converter.fasta import sequence_to_eden
from eden.modifier.rna.annotate_rna_structure import annotate_single
import subprocess as sp
import os
from tqdm import tqdm
import pdb
from multiprocessing import Queue, Lock, Process

def run_rnashape(sequence):
    #
    cmd = 'echo "%s" | RNAshapes -t %d -c %d -# %d' % (sequence, 5, 10, 1)
    out = sp.check_output(cmd, shell=True)
    text = out.strip().split('\n')
    seq_info = text[0]
    if 'configured to print' in text[-1]:
        struct_text = text[-2]
    else:
        struct_text = text[1]
    # shape:
    structur = struct_text.split()[1]
    # extract the shape bracket notation
    #shape_list += [line.split()[2] for line in struct_text]
    #encoee strucyrte
    graph = sequence_to_eden([("ID", sequence)]).next()
    graph.graph['structure']=structur
    annotate_single(graph)
    encode_struct = ''.join([x["entity_short"].upper() for x in graph.node.values() ])
    return encode_struct
    #pdb.set_trace()

def read_structure(seq_file):
    seq_list = []
    structure_list = []
    fw = open(seq_file + '.structure', 'w')
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line
                if len(seq):
                    fw.write(old_name)
                    seq = seq.replace('U', 'T')
                    struc_en = run_rnashape(seq)
                    fw.write(struc_en + '\n')
                old_name = name
                seq = ''
            else:
                seq = seq + line[:-1]
        if len(seq):
            fw.write(old_name)
            seq = seq.replace('U', 'T')
            struc_en = run_rnashape(seq)
            fw.write(struc_en + '\n')
    fw.close()

def run_predict_graphprot_data():
    data_dir = '/home/panxy/eclipse/ideep/GraphProt_CLIP_sequences/'
    #fw = open('result_file_struct_graphprot', 'w')
    for protein_file in os.listdir(data_dir):

        protein = protein_file.split('.')[0]
        print(protein)
        read_structure(data_dir + protein_file)
        #fw.write(protein + '\t')
        #model = merge_seperate_network_with_multiple_features(protein, kmer=False, rg=True, clip=True, rna=True, go=False, motif = True, seq = True, fw = fw)
        #run_individual_network(protein, kmer=False, rg=False, clip=False, rna=False, motif = False, seq = True, fw = fw)
        #run_seq_struct_cnn_network(protein, seq = True, fw= fw, graph_prot = True)
        #run_individual_network(protein, kmer=False, rg=False, clip=False, rna=False, go=False, motif = False, seq = True, oli = False, fw = fw)
    #fw.close()

def write_second(q, lock):
    while not q.empty():
        rna = q.get()
        if q.qsize() % 100 == 0:
            print(q.qsize())
        struct = run_rnashape(rna)
        lock.acquire()
        f = open('../dataset/second_strcuture', 'a')
        f.write(rna + '\t' + struct + '\n')
        f.close()
        lock.release()

if __name__ == "__main__":
    #run_predict_graphprot_data()
    # sequence = 'TGGAAACATTCCTCAGGTGGTTCATCCAAGGCCCTTTCCACTCTTTCAGCTCACAGCACAGTGGTCCTTTTGTTCTTTGGTCCACCCATGTTTGTGTATAC'
    # encode_struct = run_rnashape(sequence)
    # print(encode_struct)
    protein_list = ['AGO1', 'AGO2', 'AGO3', 'ALKBH5', 'AUF1', 'C17ORF85', 'C22ORF28', 'CAPRIN1', 'DGCR8', 'EIF4A3',
                    'EWSR1',
                    'FMRP', 'FOX2', 'FUS', 'FXR1', 'FXR2', 'HNRNPC', 'HUR', 'IGF2BP1', 'IGF2BP2', 'IGF2BP3', 'LIN28A',
                    'LIN28B',
                    'METTL3', 'MOV10', 'PTB', 'PUM2', 'QKI', 'SFRS1', 'TAF15', 'TDP43', 'TIA1', 'TIAL1', 'TNRC6',
                    'U2AF65',
                    'WTAP', 'ZC3H7B']
    dic = set()
    queue = Queue()
    lock = Lock()
    for pot in protein_list:
        fin = open('../dataset/trainset/' + pot, 'r')
        #fout = open('../dataset/trainset/' + pot + '_2nd', 'w')
        for line in tqdm(fin.readlines()):
            rna, label = line.split('\t')
            dic.add(rna)
    for r in dic:
        queue.put(str(r))
    print(len(dic))
    print(queue.qsize())
    proc = [Process(target=write_second, args=(queue, lock)) for i in range(10)]
    for p in proc:
        p.start()
    for p in proc:
        p.join()
