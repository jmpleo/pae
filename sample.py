
import sys
import os
import argparse
import numpy as np
import math
import itertools
import collections
import time

#from pcfg.lib_guesser.banner_info import print_banner
#from pcfg.lib_guesser.pcfg_grammar import PcfgGrammar
#from pcfg.lib_guesser.cracking_session import CrackingSession
#from pcfg.lib_guesser.honeyword_session import HoneywordSession
#from pcfg_guesser import create_save_config

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
    parser.add_argument(
        '--method',
        help='which model to generate',
        default=program_info['method'],
        choices=program_info['supported_methods'],
        metavar='M'
    )

    parser.add_argument(
        '--wordlist_first',
        help='first popular wordlist',
        default=program_info['pae']['wordlist_first'],
        metavar='FILE'
    )

    parser.add_argument(
        '--log_file',
        help='path to logging',
        default=program_info['log_file'],
        metavar='FILE'
    )

    ############################################################################
    ### pae
    ############################################################################
    parser.add_argument(
        '--load_model',
        default=program_info['pae']['load_model'],
        metavar='FILE',
        help='path to load checkpoint if specified'
    )

    parser.add_argument(
        '--min_len',
        type=int,
        default=program_info['pae']['min_len'],
        metavar='N',
        help='min length of generated passwords'
    )

    parser.add_argument(
        '--max_len',
        type=int,
        default=program_info['pae']['max_len'],
        metavar='N',
        help='max length of generated passwords'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=program_info['pae']['batch_size'],
        metavar='N',
        help='batch size'
    )

    parser.add_argument(
        '--similar_sample_n',
        type=int,
        default=program_info['pae']['similar_sample_n'],
        metavar='N',
        help='samples for each similar vector'
    )

    parser.add_argument(
        '--similar_std',
        help='sample similars with this standart deviation',
        default=program_info['pae']['similar_std'],
        type=float,
        metavar='STD'
    )

    parser.add_argument(
        '--alphabet',
        help='path to alphabet file',
        default=program_info['pae']['alphabet'],
        metavar='FILE'
    )

    parser.add_argument(
        '--pii',
        help='path to leaked info, (PII, passwords, nicks, etc.)',
        default=program_info['pae']['pii'],
        metavar='FILE|DIR'
    )

    parser.add_argument(
        '--stdout',
        help='print password to stdout',
        action='store_true'
    )

    parser.add_argument(
        '--save_dir',
        help='directory to save checkpoints and outputs',
        default=program_info['pae']['save_dir'],
        metavar='DIR'
    )

    ###########################################################################
    ## pcfg
    ###########################################################################

    ## Standard options for ruleset, etc
    #
    # The rule name to load the grammar from. Should be saved under the
    # 'Rules' folder. This rule needs to have been previously created by
    # the pcfg 'trainer.py' program.
    #parser.add_argument(
    #    '--rule',
    #    help = 'The ruleset to use. Default is ' +
    #    program_info['pcfg']['rule_name'],
    #    metavar = 'RULESET_NAME',
    #    required = False,
    #    default = program_info['pcfg']['rule_name']
    #)

    #parser.add_argument(
    #    '--session',
    #    help = 'Session name. Used for saving/restoring sessions Default is ' +
    #        program_info['pcfg']['session_name'],
    #    metavar = 'SESSION_NAME',
    #    required = False,
    #    default = program_info['pcfg']['session_name']
    #)

    #"""
    #parser.add_argument(
    #    '--load',
    #    '-l',
    #    help='Loads a previous guessing session',
    #    dest='load',
    #    action='store_const',
    #    const= not program_info['load_session'],
    #    default = program_info['load_session']
    #)
    #"""

    #parser.add_argument(
    #    '--limit',
    #    help='Generate N guesses and then exit. This can be used for wordlist generation and/or research evaluation',
    #    type=int,
    #    default=program_info['pcfg']['limit']
    #)

    #parser.add_argument(
    #    '--skip_brute',
    #    help='Do not perform Markov based guesses using OMEN. This is useful ' +
    #        'if you are running a seperate dedicated Markov based attack',
    #    action='store_const',
    #    const= not program_info['pcfg']['skip_brute'],
    #    default = program_info['pcfg']['skip_brute']
    #)

    #parser.add_argument(
    #    '--lowercase',
    #    help='Only generate lowercase guesses. No case mangling. (Setting is currently not applied to OMEN generated guesses)',
    #    action='store_const',
    #    const= not program_info['pcfg']['skip_case'],
    #    default = program_info['pcfg']['skip_case']
    #)

    #parser.add_argument(
    #    '--mode',
    #    help = "Method in which guesses are generated Default is '" +
    #        program_info['pcfg']['cracking_mode'] +
    #        "' Supported Modes: " + str(program_info['pcfg']['supported_modes']),
    #    metavar = 'MODE',
    #    required = False,
    #    default = program_info['pcfg']['cracking_mode'],
    #    choices = program_info['pcfg']['supported_modes']
    #)

    # Parse all the args and save them
    args=parser.parse_args()

    program_info['log_file'] = ensure_absolute_path(args.log_file, ROOT_DIR)

    program_info['pae']['load_model'] = ensure_absolute_path(args.load_model, ROOT_DIR)
    program_info['pae']['pii'] = ensure_absolute_path(args.pii, ROOT_DIR)
    program_info['pae']['min_len'] = args.min_len
    program_info['pae']['max_len'] = args.max_len
    program_info['pae']['stdout'] = args.stdout
    program_info['pae']['save_dir'] = ensure_absolute_path(args.save_dir, ROOT_DIR)
    program_info['pae']['batch_size'] = args.batch_size
    program_info['pae']['alphabet'] = ensure_absolute_path(args.alphabet, ROOT_DIR)
    program_info['pae']['similar_sample_n'] = args.similar_sample_n
    program_info['pae']['similar_std'] = args.similar_std
    program_info['pae']['wordlist_first'] = ensure_absolute_path(args.wordlist_first, ROOT_DIR)


    # Standard Options
    #program_info['pcfg']['rule_name'] = args.rule
    #program_info['pcfg']['session_name'] = args.session
    ##program_info['pcfg']['load_session'] = args.session

    ## Advanced Options
    #program_info['pcfg']['limit'] = args.limit
    #program_info['pcfg']['skip_brute'] = args.skip_brute
    #program_info['pcfg']['skip_case'] = args.lowercase
    #program_info['pcfg']['cracking_mode'] = args.mode

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
            'min_len': 5,
            'max_len': 20,

            'batch_size': 1000,

            'load_model': 'out/old/models/laae-0.01.pt',
            'alphabet': 'out/old/models/vocab.alphabet',

            'pii': None,

            'wordlist_first': None,

            'similar_sample_n': 10,
            'similar_std': 0.1,

            'save_dir': 'samples'
        },

        #'pcfg': {
        #    # Program and Contact Info
        #    'name':'PCFG Guesser',
        #    'version': '4.6',
        #    'author':'Matt Weir',
        #    'contact':'cweir@vt.edu',

        #    # Standard Options
        #    'rule_name':'top1kk',
        #    'session_name':'default_run',
        #    'load_session':False,
        #    'limit': None,

        #    # Cracking Mode options
        #    'cracking_mode':'true_prob_order',
        #    'supported_modes':['true_prob_order', 'random_walk', 'honeywords'],

        #    # Advanced Options
        #    'skip_brute': False,
        #    'skip_case': False
        #}
    }

    args = parse_command_line(program_info)

    ############################################################################
    ## path setup
    ############################################################################
    requred_dirs = [
        #"pcfg",
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

        session = SessionPAE(
            load_model       = program_info['pae']['load_model'],
            pii_load         = program_info['pae']['pii'],
            similar_std      = program_info['pae']['similar_std'],
            similar_sample_n = program_info['pae']['similar_sample_n'],
            min_len          = program_info['pae']['min_len'],
            max_len          = program_info['pae']['max_len'],
            save_dir         = program_info['pae']['save_dir'],
            log_file         = log_file,
            batch_size       = program_info['pae']['batch_size'],
            alphabet         = program_info['pae']['alphabet'],
            stdout           = program_info['pae']['stdout'],
            wordlist         = program_info['pae']['wordlist_first']
        )

    #elif args.method == 'pcfg':

    #    base_directory = os.path.join(
    #        root_dir,
    #        'pcfg',
    #        'Rules',
    #        program_info['pcfg']['rule_name']
    #    )

    #    session_save = os.path.join(
    #        root_dir,
    #        '' + program_info['pcfg']['session_name'] + '.sav'
    #    )

    #    save_config = create_save_config(program_info['pcfg'])

    #    pcfg = PcfgGrammar(
    #        program_info['pcfg']['rule_name'],
    #        base_directory,
    #        program_info['pcfg']['version'],
    #        session_save,
    #        skip_brute = program_info['pcfg']['skip_brute'],
    #        skip_case = program_info['pcfg']['skip_case']
    #        #debug = False #program_info['debug']
    #    )

    #    session = CrackingSession(pcfg, save_config, session_save)

    else:
        logging(log_file, "Exiting...")
        return 1

    session.run()

    #if total_passwords > args.limit:
    #    logging("# WARNING! the specified limit is small for the specified PII and lengths",
    #            log_file)


if __name__ == '__main__':
    main()



