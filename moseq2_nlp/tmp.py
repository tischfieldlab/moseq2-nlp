import configargparse

parser = configargparse.ArgParser(default_config_files=['./tmp_config.cfg'])

parser.add_argument('--x', action='append', nargs='?')
args = parser.parse_args()

if args.x is not None:
    tmp = [s.split(',') for s in args.x]
else:
    tmp = [0,1,2,3]
print(tmp)
