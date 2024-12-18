
import sys
import os
import argparse
import numpy as np
import math
import itertools
import collections
import time

from huggingface_hub import hf_hub_download

from pae_guesser import SessionPAE

from utils import logging, dump


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


def ensure_absolute_path(path, root_dir):
    if path is None:
        return None

    return os.path.abspath(path) if os.path.isabs(path) else os.path.join(root_dir, path)


def parse_command_line(program_info):

    parser = argparse.ArgumentParser(
        description=program_info['name']
    )

    ############################################################################
    ### global
    ############################################################################
    parser.add_argument('--method',
                                    help='which model to generate',
                                    default=program_info['method'],
                                    choices=program_info['supported_methods'],
                                    metavar='M')

    parser.add_argument('--wordlist_first',
                                            help='first popular wordlist',
                                            default=program_info['pae']['wordlist_first'],
                                            metavar='FILE')

    parser.add_argument('--log_file',
                                        help='path to logging',
                                        default=program_info['log_file'],
                                        metavar='FILE')

    parser.add_argument('--save_dir',
                                    help='directory to save checkpoints and outputs',
                                    default=program_info['pae']['save_dir'],
                                    metavar='DIR')

    ############################################################################
    ### pae
    ############################################################################
    parser.add_argument('--load_model',
                                        default=program_info['pae']['load_model'],
                                        metavar='PATH',
                                        help='path to load checkpoint if specified')

    parser.add_argument('--repo_id',
                                        default=program_info['pae']['repo_id'],
                                        metavar='REPO',
                                        help='path to repo src of model')

    parser.add_argument('--local',
                                    help='using load_model arg as local path',
                                    action='store_true')

    # todo:
    # parser.add_argument('--len', 
    #                            type=str,
    #                            default=program_info['pae']['len'],
    #                            metavar='MIN',
    #                            help='length of generated passwords (same as "8";"8-10";"8,9,11")')

    # parser.add_argument('--min_len',
    #                                 type=int,
    #                                 default=program_info['pae']['min_len'],
    #                                 metavar='MIN',
    #                                 help='min length of generated passwords')

    # parser.add_argument('--max_len',
    #                                 type=int,
    #                                 default=program_info['pae']['max_len'],
    #                                 metavar='MAX',
    #                                 help='max length of generated passwords')

    parser.add_argument('--batch_size',
                                    type=int,
                                    default=program_info['pae']['batch_size'],
                                    metavar='B',
                                    help='batch size')

    parser.add_argument('--sigmas_n',
                                    type=int,
                                    default=program_info['pae']['sigmas_n'],
                                    metavar='N',
                                    help='pints of sigmas on [sigma_min, sigma_max]')

    parser.add_argument('--sigma_min',
                                        help='min sigma sampling',
                                        default=program_info['pae']['sigma_min'],
                                        type=float,
                                        metavar='SMIN')  
    
    parser.add_argument('--sigma_max',
                                        help='max sigma sampling',
                                        default=program_info['pae']['sigma_max'],
                                        type=float,
                                        metavar='SMAX')

    #parser.add_argument(
    #    '--alphabet',
    #    help='path to alphabet file',
    #    default=program_info['pae']['alphabet'],
    #    metavar='FILE'
    #)

    parser.add_argument('--pii',
                                help='path to leaked info, (PII, passwords, nicks, etc.)',
                                default=program_info['pae']['pii'],
                                metavar='FILE|DIR')

    parser.add_argument('--stdout',
                                    help='print password to stdout',
                                    action='store_true')


    parser.add_argument("--cuda",
                                action="store_true",
                                help="disable CUDA")



    # Parse all the args and save them
    args=parser.parse_args()

    program_info['log_file'] = ensure_absolute_path(args.log_file, ROOT_DIR)

    program_info['pae']['load_model'] = args.load_model
    program_info['pae']['repo_id'] = args.repo_id
    program_info['pae']['local'] = args.local

    program_info['pae']['pii'] = ensure_absolute_path(args.pii, ROOT_DIR)
    #program_info['pae']['min_len'] = args.min_len
    #program_info['pae']['max_len'] = args.max_len
    program_info['pae']['stdout'] = args.stdout
    program_info['pae']['save_dir'] = ensure_absolute_path(args.save_dir, ROOT_DIR)
    program_info['pae']['batch_size'] = args.batch_size
    #program_info['pae']['alphabet'] = ensure_absolute_path(args.alphabet, ROOT_DIR)
    program_info['pae']['sigmas_n'] = args.sigmas_n
    program_info['pae']['sigma_min'] = args.sigma_min
    program_info['pae']['sigma_max'] = args.sigma_max
    program_info['pae']['wordlist_first'] = ensure_absolute_path(args.wordlist_first, ROOT_DIR)
    program_info['pae']['cuda'] = args.cuda

    args = parser.parse_args()

    return args


def main():
    """
        Main function
    """
    program_info = {
        'name': 'sample autogen',
        'method': 'pae',
        'supported_methods': ['pae'], # todo: 'pcfg', 'passgan'],
        'log_file': 'log.txt',
        'pae' : {
            #'min_len': 8,
            #'max_len': 8,
            'batch_size': 4096,
            'repo_id': 'jmpleo/pae',
            'load_model': 'v1/rand-10-12-14-30000k/laae-0.01/model.pt',
            'pii': None,
            'wordlist_first': None,
            'sigmas_n': 50,
            'sigma_min': 0.001,
            'sigma_max': 0.3,
            'save_dir': 'samples'
        }
    }

    args = parse_command_line(program_info)

    ############################################################################
    ## path setup
    ############################################################################
    requred_dirs = [
        "pae"
    ]

    log_file = program_info['log_file']

    logging(log_file, str(args))

    if not all(os.path.exists(os.path.join(ROOT_DIR, folder)) for folder in requred_dirs):

        logging(
            log_file,
            f"WARNING: not found requred dirs: "
            f"{', '.join(path for path in requred_dirs)}"
        )

        for folder in requred_dirs:
            os.makedirs(os.path.join(ROOT_DIR, folder), exist_ok=True)


    ############################################################################
    ## guesser session
    ############################################################################
    if args.method == 'pae':


        if not os.path.exists(program_info['pae']['save_dir']):
            os.makedirs(program_info['pae']['save_dir'])

        if program_info['pae']['local']:
            model_path = program_info['pae']['load_model']
        
        else:
            model_path = hf_hub_download(
                program_info['pae']['repo_id'], 
                program_info['pae']['load_model']
            )

        session = SessionPAE(
            model_path       = model_path,
            pii_load         = program_info['pae']['pii'],
            sigma_min        = program_info['pae']['sigma_min'],
            sigma_max        = program_info['pae']['sigma_max'],
            sigmas_n         = program_info['pae']['sigmas_n'],
            #min_len          = program_info['pae']['min_len'],
            #max_len          = program_info['pae']['max_len'],
            save_dir         = program_info['pae']['save_dir'],
            log_file         = log_file,
            batch_size       = program_info['pae']['batch_size'],
            #alphabet         = program_info['pae']['alphabet'],
            stdout           = program_info['pae']['stdout'],
            wordlist         = program_info['pae']['wordlist_first'],
            cuda             = program_info['pae']['cuda']
        )

    else:
        logging(log_file, "Exiting...")
        return 1

    session.run()

    #if total_passwords > args.limit:
    #    logging("# WARNING! the specified limit is small for the specified PII and lengths",
    #            log_file)


if __name__ == '__main__':
    main()



