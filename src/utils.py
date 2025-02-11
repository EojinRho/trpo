"""
Logging and Data Scaling Utilities

Written by Patrick Coady (pat-coady.github.io)
"""
import numpy as np
import json
import os
import shutil
import glob
import csv


class Scaler(object):
    """ Generate scale and offset based on running mean and stddev along axis=0

        offset = running mean
        scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, obs_dim):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.m = 0
        self.n = 0
        self.first_pass = True

    def update(self, x):
        """ Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)

        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.m += n

    def get(self):
        """ returns 2-tuple: (scale, offset) """
        return 1/(np.sqrt(self.vars) + 0.1)/3, self.means


class Logger(object):
    """ Simple training logger: saves to file and optionally prints to stdout """
    def __init__(self, logname, now):
        """
        Args:
            logname: name for log (e.g. 'Hopper-v1')
            now: unique sub-directory name (e.g. date/time string)
        """
        path = os.path.join('log-files', logname, now)
        self.base_path = path
        self.model_count = 0
        self.model = None
        self.model_2 = None
        self.model_3 = None
        self.env_name = logname
        os.makedirs(path)
        filenames = glob.glob('*.py')  # put copy of all python files in log_dir
        for filename in filenames:     # for reference
            shutil.copy(filename, path)
        path = os.path.join(path, 'log.csv')

        self.write_header = True
        self.log_entry = {}
        self.f = open(path, 'w')
        self.writer = None  # DictWriter created with first call to write() method

    def write(self, display=True):
        """ Write 1 log entry to file, and optionally to stdout
        Log fields preceded by '_' will not be printed to stdout

        Args:
            display: boolean, print to stdout
        """
        if display:
            print(self.env_name)
            print(os.path.join(self.base_path, 'model.'+str(self.model_count)+'.json'))
            self.disp(self.log_entry)
        if self.write_header:
            fieldnames = [x for x in self.log_entry.keys()]
            self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
            self.writer.writeheader()
            self.write_header = False

        # write model
        if self.model and self.model_count % 25 == 0:
            model_file_path = os.path.join(self.base_path, 'model.'+str(self.model_count)+'.json')
            mf = open(model_file_path, 'wt')
            json.dump(self.model, mf, indent=0, separators=(',', ':'))
            mf.close()

            model_file_path_2 = os.path.join(self.base_path, 'value.'+str(self.model_count)+'.json')
            mf = open(model_file_path_2, 'wt')
            json.dump(self.model_2, mf, indent=0, separators=(',', ':'))
            mf.close()

            if self.model_3 != None:
                model_file_path_3 = os.path.join(self.base_path, 'auto.csv')
                mf = open(model_file_path_3, 'a')
                mf.write("%d,%f,%f,%f,%f" % (self.model_count,  self.model_3["alive_coef"], self.model_3["progress_coef"],
                                                                self.model_3["alive_sum"], self.model_3["progr_sum"]))
                mf.close()

        self.model_count += 1
        self.model = None

        self.writer.writerow(self.log_entry)
        self.log_entry = {}
        self.model = None
        self.model_2 = None
        self.model_3 = None

    @staticmethod
    def disp(log):
        """Print metrics to stdout"""
        log_keys = [k for k in log.keys()]
        log_keys.sort()
        print('***** Episode {}, Mean R = {:.1f}, Adv = {:.2f}, Adv_Min = {:.2f} Adv_Max = {:.2f} *****'.format(log['_Episode'],
                            log['_MeanReward'], log['_mean_adv'], log['_min_adv'], log['_max_adv']))
        for key in log_keys:
            if key[0] != '_':  # don't display log items with leading '_'
                print('{:s}: {:.3g}'.format(key, log[key]))
        print('\n')

    def log(self, items):
        """ Update fields in log (does not write to file, used to collect updates.

        Args:
            items: dictionary of items to update
        """
        self.log_entry.update(items)

    def log_model(self, model_list):
        """ stores the model (as a python list of names and param values)

        Args:
            model_list: list of param names and values
        """
        self.model = model_list

    def log_model_2(self, model_list):
        """ stores the model (as a python list of names and param values)

        Args:
            model_list: list of param names and values
        """
        self.model_2 = model_list

    def log_model_3(self, model_list):
        """ stores the model (as a python list of names and param values)

        Args:
            model_list: list of param names and values
        """
        self.model_3 = model_list

    def close(self):
        """ Close log file - log cannot be written after this """
        self.f.close()
