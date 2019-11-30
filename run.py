from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import numpy as np

from FAERSdata import FAERSdata
from Model import Model
from utils import split_data, sample_zeros


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--input', required=True, help='Input original signal scores file.')
    parser.add_argument('--method', required=True, choices=['PRR05', 'ROR05', 'GPS', 'BCPNN'], help='Signal detection algorithm')
    parser.add_argument('--year', default='all', choices=['all', 'each'], help='Years of data used for model')
    parser.add_argument('--eval_metrics', required=True, choices=['all', 'specificity-sensitivity'],
                        help='Evaluation metrics')
    parser.add_argument('--split', type=bool, default=False)
    parser.add_argument('--output')

    args = parser.parse_args()
    return args


def pretty_print_eval(res, metrics):
    if metrics == 'all':
        print('All metrics: ' + ','.join(np.round(res,3).astype(str)))
    else:
        print('fixed_sensitivity: ' + ','.join(np.round(res[1],3).astype(str)))
        print('fixed_specificity: ' + ','.join(np.round(res[2],3).astype(str)))


def main(args):
    print('#' * 50)
    print('Signal Detection Algorithm: {}, Year: {}'.format(args.method, args.year))
    print('#' * 50)


    data = FAERSdata(args.input, args.method, args.year)

    for i in range(len(data.X.keys())):
        X, Y, _ = data.X.get(i), data.Y.get(i), data.Index.get(i)
        # all_idx = np.where(Y > -1)
        all_idx = sample_zeros(Y)
        if args.split:
            valid, test = split_data(Y)
            model = Model(args.eval_metrics)
            model.validate(X, Y, valid)
            Y_pred = model.predict(X, model.ALPHA)
            valid_res = model.eval(Y_pred, Y, valid)
            test_res = model.eval(Y_pred, Y, test)
            print('LP-{}:'.format(args.method))
            print('alpha: {}'.format(model.ALPHA))
            print('valid:')
            pretty_print_eval(valid_res, args.eval_metrics)
            print('test:')
            pretty_print_eval(test_res, args.eval_metrics)

            valid_res = model.eval(X, Y, valid)
            test_res = model.eval(X, Y, test)
            print('baseline-{}:'.format(args.method))
            print('valid:')
            pretty_print_eval(valid_res, args.eval_metrics)
            print('test:')
            pretty_print_eval(test_res, args.eval_metrics)
        else:
            model = Model(args.eval_metrics)
            model.validate(X, Y, all_idx)
            Y_pred = model.predict(X, model.ALPHA)
            res = model.eval(Y_pred, Y, all_idx)
            print('LP-{}:'.format(args.method))
            pretty_print_eval(res, args.eval_metrics)

            print('baseline-{}:'.format(args.method))
            res = model.eval(X, Y, all_idx)
            pretty_print_eval(res, args.eval_metrics)

def main_DME(args):
    print('#' * 50)
    print('Signal Detection Algorithm: {}, Year: {}'.format(args.method, args.year))
    print('#' * 50)

    data = FAERSdata(args.input, args.method, args.year)
    DME = np.loadtxt('DME.txt', dtype=str, delimiter=',')
    adr_id, adr_name = DME[:,0], DME[:,1]

    out = open(args.output, 'w')
    # out.write('ID,Name,AUC,AUC,AUPR,AUPR,Precision,Precision,Recall,Recall,Accuracy,Accuracy,F1,F1\n')
    for i in range(len(data.X.keys())):
        X, Y, _ = data.X.get(i), data.Y.get(i), data.Index.get(i)
        # all_idx = np.where(Y > -1)
        eval_idx = sample_zeros(Y)
        model = Model(args.eval_metrics)
        Y_pred = model.predict(X, model.ALPHA)
        LP_res = model.eval_DME(Y_pred, Y, eval_idx, adr_id)
        baseline_res = model.eval_DME(X, Y, eval_idx, adr_id)
        for i, adr in enumerate(list(adr_id)):
            print('LP-{}:'.format(args.method))
            LP_metric = LP_res.get(adr)
            print('ADR:{} '.format(adr))
            pretty_print_eval(LP_metric, args.eval_metrics)

            print('baseline-{}:'.format(args.method))
            baseline_metric = baseline_res.get(adr)
            pretty_print_eval(baseline_metric, args.eval_metrics)

            out.write('{},{},{},{}\n'.format(adr, adr_name[i], ','.join(np.round(LP_metric,3).astype(str)), ','.join(np.round(baseline_metric,3).astype(str))))

    out.close()


def more_main():
    args = parse_args()
    main(args)

if __name__ == '__main__':
    more_main()
