from .Bert import BertForLinkPredict
from .BertSage import BertMaxSageForLinkPredict, BertMeanSageForLinkPredict, BertSumSageForLinkPredict
from .BertGAT import BertGATForLinkPredict
from .Graphformer import GraphFormersForLinkPredict

AutoModels = {
    'bert': BertForLinkPredict,
    'bert-maxsage': BertMaxSageForLinkPredict,
    'bert-meansage': BertMeanSageForLinkPredict,
    'bert-sumsage': BertSumSageForLinkPredict,
    'bert-gat': BertGATForLinkPredict,
    'graphformer': GraphFormersForLinkPredict
}
