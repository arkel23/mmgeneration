import ast
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_log_json', type=str, help='log json file')
    parser.add_argument('--metric', type=str, default='fid', help='metric')
    parser.add_argument('--max', action='store_true', help='max (min default)')
    args = parser.parse_args()

    d_i_m = {}
    with open(args.path_log_json) as f:
        lines = f.readlines()
        for line in lines:
            if '"mode": "val"' in line:
                temp = ast.literal_eval(line)
                d_i_m[f"{temp['iter']}"] = temp[f'{args.metric}']

    if args.max:
        best_value = max(d_i_m, key=d_i_m.get)
    else:
        best_value = min(d_i_m, key=d_i_m.get)
    print(f'Best iter: {best_value}\t {args.metric}: {d_i_m[best_value]}')
    last_key = list(d_i_m)[-1]
    print(f'Last iter: {last_key}\t {args.metric}: {d_i_m[last_key]}')


main()
