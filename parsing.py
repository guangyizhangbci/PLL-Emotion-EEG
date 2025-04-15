import argparse


def create_parser():
    parser = argparse.ArgumentParser(description='Partial Label Learning for Emotion Recognition from EEG')

    # Genreal
    parser.add_argument('--method',         default='DNPL', type=str,               help='method name')
    parser.add_argument('--num-class',      default=5,      type=int,               help='Number of emotion classes')
    parser.add_argument('--lr',             default=0.01,   type=float,             help='initial learning rate')
    parser.add_argument('--epochs',         default=30,     type=int,               help='epochs')
    parser.add_argument('--batch-size',     default=8,      type=int,               help='batch size')
    parser.add_argument('--optimizer',      default='sgd',  type=str,               help='optimizer choice')
    parser.add_argument('--use-scheduler',  default=False,  action="store_true",    help='learning rate scheduler')
    parser.add_argument('--run-idx',        default=1,      type=int,               help='repeat with indepedent random seed')
    parser.add_argument('--use-confidence', default=False,  action="store_true",    help='label confidence ones initialization')
    parser.add_argument('--partial-type',   default='uniform', type=str,           help='partial label type')

    # LW
    parser.add_argument('--beta',           default=0.0,    type = float,           help='weight of loss applied on non-candiate labels')
    parser.add_argument('--loss',           default='cross_entropy',type = str,     help='LW-loss, sigmoid or cross-entropy')


    # CR
    parser.add_argument('--lam',            default=1.0,    type=float,             help='weight of consistency loss')
    parser.add_argument('--c-weight',       default=1.0,    type=float,             help='consistency_weight for original')
    parser.add_argument('--c-weight-w',     default=1.0,    type=float,             help='consistency_weight for weak augmentation')
    parser.add_argument('--c-weight-s',     default=1.0,    type=float,             help='consistency_weight for strong augmentation')



    # PiCO

    parser.add_argument('--gamma',          default=0.5,    type = float,           help='contrastive loss weight')
    parser.add_argument('--low-dim',        default=64,     type=int,               help='embedding dimension')
    parser.add_argument('--moco_queue',     default=1000,   type=int,               help='queue size; number of negative samples')
    parser.add_argument('--moco_m',         default=0.999,  type=float,             help='momentum for updating momentum encoder')
    parser.add_argument('--proto_m',        default=0.99,   type=float,             help='momentum for computing the momving average of prototypes')
    parser.add_argument('--momentum',       default=0.9,    type=float,             help='momentum of SGD solver')
    parser.add_argument('--weight-decay',   default=1e-5,   type=float,             help='weight decay (default: 1e-5)')
    parser.add_argument('--cosine',         default=False,  action='store_true',    help='use cosine lr schedule')
    parser.add_argument('--conf-ema-range', default='0.95,0.8', type=str,           help='pseudo target updating coefficient (phi)')
    parser.add_argument('-lr_decay_epochs', default='700,800,900', type=str,        help='where to decay lr, can be a list')


    # Emotion setting
    parser.add_argument('--emotion-delta',  default=0.0, type=float,  help='scale the simlarity score, in order to compare with classical labels generation in terms of ambiguity')
    
    
    return parser
