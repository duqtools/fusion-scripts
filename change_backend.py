from packaging import version
import os
from os import path
import inspect
import types
import getpass
import sys
import argparse

import xml.sax
import xml.sax.handler

min_imas_version_str = "3.28.0"
min_imasal_version_str = "4.7.2"

try:
    import imas
except ImportError:
    warnings.warn("IMAS Python module not found or not configured properly, tools need IDS to work!", UserWarning)
if imas is not None:
    from imas import imasdef
    vsplit = imas.names[0].split("_")
    imas_version = version.parse(".".join(vsplit[1:4]))
    ual_version = version.parse(".".join(vsplit[5:]))
    if imas_version < version.parse(min_imas_version_str):
        raise ImportError("IMAS version must be >= %s! Aborting!" % (min_imas_version_str))
    if ual_version < version.parse(min_imasal_version_str):
        raise ImportError("IMAS AL version must be >= %s! Aborting!" % (min_imasal_version_str))

def input():

    parser = argparse.ArgumentParser(
    description=
    """Changes the backend of an IDS
    ---
    Examples:\n
    python change_backend.py --backend 'hdf5' --database tcv --version 3 --shot 55592 --run 1 \n
    ---
    """,
    epilog="",
    formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--backend",    "-b",        type=str,   default="hdf5",    choices=["mdsplus", "hdf5"],       help="Backend with which to access data")
    parser.add_argument("--database",   "-i",        type=str,   default=None,                                         help="name of the database where data is stored")
    parser.add_argument("--version",    "-v",        type=str,   default="3",                                          help="UAL version")
    parser.add_argument("--shot",                    type=int,   default=None,                                         help="Shot number")
    parser.add_argument("--run",                     type=int,   default=None,                                         help="Run number")
    args=parser.parse_args()

    return args



def change_backend_single(db, shotnumber, run, ids_list = [], backend = 'hdf5'):

    username = None
    if not username: username = getpass.getuser()

    if backend == 'hdf5':

        path = '/afs/eufus.eu/user/g/' + username + '/public/imasdb/tcv/3/0/'
        if run < 10:
            run_str = '000' + str(run)
        elif run < 100:
            run_str = '00' + str(run)
        elif run < 1000:
            run_str = '0' + str(run)
        else:
            run_str = str(run)

        filename = path + 'ids_' + str(shotnumber) + run_str + '.datafile'

    elif backend == 'mdsplus':
        path = '/afs/eufus.eu/user/g/' + username + '/public/imasdb/tcv/3/' + str(shotnumber) + '/' + str(run) + '/'

        filename = path + 'master.h5'

    if os.path.isfile(filename):
        copy_ids_entry(db, shotnumber, run, shotnumber, run, backend = backend)


def change_backend_all(db, shotlist, ids_list = [], backend = 'hdf5'):

    username = None
    if not username: username = getpass.getuser()

    path = '/afs/eufus.eu/user/g/' + username + '/public/imasdb/tcv/3/0/'

    for shot in shotlist:
        for run in range(0, 10000):
            if run < 10:
                run_str = '000' + str(run)
            elif run < 100:
                run_str = '00' + str(run)
            elif run < 1000:
                run_str = '0' + str(run)
            else:
                run_str = str(run)

            filename = path + 'ids_' + str(shot) + run_str + '.datafile'
            if os.path.isfile(filename):
                copy_ids_entry(db, shot, run, shot, run, backend = backend)


class Parser(xml.sax.handler.ContentHandler):
    def __init__(self):
        xml.sax.handler.ContentHandler.__init__(self)
        self.idss = []

    def startElement(self, name, attrs):
        if name == 'IDS':
            ids = dict()
            for i in attrs.getNames():
                ids[i] = attrs.getValue(i)
            self.idss.append(ids)

def copy_ids_entry(db, shot, run, shot_target, run_target, ids_list = [], username = None, backend = 'hdf5'):

    '''

    Copies an entire IDS entry

    '''

    if not username:
        username = getpass.getuser()

    # open input pulsefile and create output one

# path hardcoded for now, not ideal but avoids me to insert the version everytime. Might improve later
    path = '/gw/swimas/core/installer/src/3.34.0/ual/4.9.3/xml/IDSDef.xml'
    parser = Parser()
    xml.sax.parse(path, parser)

    vsplit = imas.names[0].split("_")
    imas_version = version.parse(".".join(vsplit[1:4]))
    imas_major_version = str(imas_version)[0]
    ual_version = version.parse(".".join(vsplit[5:]))

    print('Opening', username, db, imas_version, shot, run)
    idss_in = imas.ids(shot, run)

    if backend == 'hdf5':
        op = idss_in.open_env_backend(username, db, imas_major_version, imasdef.MDSPLUS_BACKEND)
    if backend == 'mdsplus':
        op = idss_in.open_env_backend(username, db, imas_major_version, imasdef.HDF5_BACKEND)

    if op[0]<0:
        print('The entry you are trying to copy does not exist')
        exit()
    print('Creating', username, db, imas_version, shot_target, run_target)

    #idss_out = imas.ids(shot_target, run_target)
    #idss_out.create_env(username, db, imas_major_version)

    if backend == 'mdsplus':
        idss_out = imas.DBEntry(imasdef.MDSPLUS_BACKEND, 'tcv', shot_target, run_target)
    if backend == 'hdf5':
        idss_out = imas.DBEntry(imasdef.HDF5_BACKEND, 'tcv', shot_target, run_target)
    idx = idss_out.create()[1]
    #idx = idss_out.expIdx
    #ids_list = None
    # read/write every IDS

    for ids_info in parser.idss:
        name = ids_info['name']
        maxoccur = int(ids_info['maxoccur'])
        if ids_list and name not in ids_list:
            continue
        if name == 'ec_launchers' or name == 'numerics' or name == 'sdn' or name == 'wall':
#        if name == 'ec_launchers' or name == 'numerics' or name == 'sdn' or name == 'nbi':    # test for nbi ids, temporary
            continue
            print('continue on ec launchers')  # Temporarily down due to a malfunctioning of ec_launchers ids
            print('skipping numerics')  # Not in the newest version of IMAS
        for i in range(maxoccur + 1):
            if not i:
                print('Processing', ids_info['name'])
#            if i:
#                print('Processing', ids_info['name'], i)
#            else:
#                print('Processing', ids_info['name'])

            ids = idss_in.__dict__[name]
            stdout = sys.stdout

            sys.stdout = open('/afs/eufus.eu/user/g/' + username + '/warnings_imas.txt', 'w') # suppress warnings
            ids.get(i)
#            sys.stdout = stdout
            ids.setExpIdx(idx)
            ids.put(i)
            sys.stdout.close()
            sys.stdout = stdout

    idss_in.close()
    idss_out.close()

def main():

    args = input()

    database = args.database
    shot = args.shot
    run = args.run
    backend = args.backend

    change_backend_single(database, shot, run, backend = backend)

if __name__ == "__main__":
    main()

