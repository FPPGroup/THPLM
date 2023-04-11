#!/usr/bin/env python3

import argparse
import pathlib
from torch.utils.data import DataLoader
import torch,sys,os
import numpy as np
import pandas as pd
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer


def create_parser():
    script_dir = os.path.split(os.path.realpath(__file__))[0]
    parser = argparse.ArgumentParser(
        description="Extract mean representations and model outputs for sequences in a FASTA file and to predict DDGs"  # noqa
    )

    parser.add_argument(
        "variants_file",
        type=pathlib.Path,
        default='%s/examples/var.txt'%script_dir,
        help="files inclusing variants, format is <wildtype><position><mutation> (see README for models)",
    )
    # parser.add_argument(
    #     "--model_location",
    #     type=str,
    #     help="PyTorch model file OR name of pretrained model to download ",
    # )
    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        default='%s/examples/wild.fasta'%script_dir,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        default="%s/examples/esm3Bout/"%script_dir,
        help="output directory for extracted representations",
    )
    parser.add_argument(
        "variants_fasta_dir",
        type=pathlib.Path,
        default='%s/examples/varlist.fasta'%script_dir,
        help="FASTA file was used to store variant and wildtype protein",
    )
    parser.add_argument(
        "--gpunumber",
        type=int,
        default=0,
        help="GPU number for use",
    )
    parser.add_argument("--toks_per_batch", type=int, default=4096, help="maximum batch size")
    parser.add_argument("--THPLM", type=pathlib.Path, default="%s/Model/THPLM.pt"%script_dir, help="maximum batch size")
    parser.add_argument("--extractfile", type=pathlib.Path, default='%s/esmscripts/extract.py'%script_dir, help="the path of extract.py file from esm-2")
    parser.add_argument(
        "--pythonbin",
        type=str,
        default='python',
        help="the path of python bin",
    )
    #parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser

class MyCNNNet(torch.nn.Module):
    def __init__(self):
        super(MyCNNNet, self).__init__()
        self.cnn_layers = torch.nn.Sequential(

            torch.nn.Conv1d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),
            #torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(8, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            #torch.nn.MaxPool1d(2),

        )


        self.age_fc_layers = torch.nn.Sequential(
            torch.nn.Linear(2560*32, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size()[0], -1)
        #print(x.shape)
        x= self.age_fc_layers(x)
        return x


def parser_rep(args):

    #identify file or path

    #parser input
    #parser variant list

    varlist = []
    try:
        with open(args.variants_file,'r') as varfile:
            for eachline in varfile:
                varlist.append(eachline.strip('\n'))
    except FileNotFoundError as e:
        print('variants file does not avaliable!')
    except IOError as e:
        print('The file can not open!')

    #parser wildtype protein
    variantsfile = open(args.variants_fasta_dir,'w')
    seq = ''
    with open(args.fasta_file) as variant_fasta:
        for varline in variant_fasta:
            if len(varline.strip('\n')) != 0:
                if varline[0] != '>':
                    seq += varline.strip('\n')
                else:
                    IDs = varline.strip('\n')[1:]
                    variantsfile.write(varline)
        variantsfile.write(seq +'\n')

    #parser to sequences and check
    varrep = []
    for varwpm in varlist:
        wildAA = varwpm[0]
        mutaAA = varwpm[-1]
        posi = int(varwpm[1:-1])
        if seq[posi-1] == wildAA:
            newseq = seq[:posi-1] + mutaAA + seq[posi:]
            variantsfile.write('>' + IDs + '_' + varwpm + '\n')
            varrep.append(IDs + '_' + varwpm)
            variantsfile.write(newseq + '\n')
            newseq=''
        else:
            print('The mutation {} is wrong, Please input right mutation!'.format(varwpm))
            sys.exit(0)
#/public/data1/gongj/esm/scripts/extract.py
    #run esm-2
    cmd = 'CUDA_VISIBLE_DEVICES={} time {} {} esm2_t36_3B_UR50D {}  {}/ --repr_layers 36 --include mean --toks_per_batch {}'.format(args.gpunumber,args.pythonbin,args.extractfile,args.variants_fasta_dir,args.output_dir,args.toks_per_batch)
    #print(cmd)
    os.system(cmd)
    ######### repe extract ############
    queryproteinrep = []
    for eachvar in varrep:
        rep_changes = torch.load(os.path.join(args.output_dir,eachvar+'.pt'))['mean_representations'][36]-torch.load(os.path.join(args.output_dir,eachvar.split('_')[0]+'.pt'))['mean_representations'][36]
        queryproteinrep.append(rep_changes.tolist())

    return torch.Tensor(queryproteinrep).unsqueeze(1),varrep

def main(args):
    query_rep,varrep = parser_rep(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpunumber)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('the device is %s' % device)

    model = MyCNNNet().to(device)
    model.load_state_dict(torch.load(args.THPLM))
    model.eval()
    output = model(query_rep.to(device))
    output = output.cpu().detach().numpy().reshape(-1, )
    DDG = {}
    for eachi in range(len(varrep)):
        DDG.update({varrep[eachi]:output[eachi]})
    return DDG

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    output = main(args)
    print(output)
