import numpy as np
import pylab as pl
import sciris as sc
import covasim as cv
import scipy as sp
import optuna as op
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

'''
Class for calibratopn process via Optuna 
(read tutorial in Covasim documentation https://docs.idmod.org/projects/covasim/en/latest/tutorials/t7.html and Optuna documentation https://optuna.readthedocs.io/en/stable/https://optuna.readthedocs.io/en/stable/)
'''

class Calibration:


    def __init__(self, popdict, storage, pdict, location, pop_location, start_day, end_day, 
                 datafile, cal, cal_keys, cal_values, n_trials=100, n_workers=1, n_runs=7,
                 school_days=None, school_changes=None, to_plot = ['new_diagnoses',  'new_deaths', 'n_critical']):
        '''
        Args:
            storage - storage for Optuna
            pdict - dictionary with parameters bounds
            location - city/region/country (string)
            pop_location - population in location
            start_day - start day of modellibg
            end_day - end day of modelling
            datafile - smoothed statistics
            cal - list with previous calibrated parameters
            cal_keys - which statistics to include in functional
            cal_values - weights for cal_keys
            school_days, school_changes - for school intervention (if provided)
        '''
        # Settings
        self.pop_size = pop_location # Number of agents
        self.location=location
        self.pop_location=pop_location
        self.start_day = start_day
        self.end_day = end_day
        self.datafile = datafile
        self.popdict = popdict


        # Saving and running
        self.n_trials  = n_trials # Number of sequential Optuna trials
        self.n_workers = n_workers # Number of parallel Optuna threads -- incompatible with n_runs > 1
        self.n_runs    = n_runs # Number of sims being averaged together in a single trial -- incompatible with n_workers > 1
        self.storage   = storage # Database location
        self.name      = 'covasim' # Optuna study name -- not important but required
        
        # For school interventions
        self.school_days=school_days
        self.school_changes=school_changes 
        
        # For calibration
        self.cal_keys=cal_keys        # keys of calibrated statistics (dict)
        self.cal_values=cal_values    # values of calibrated statistics (dict)
        self.pdict = pdict            # bounds for parameters
        self.cal = cal                # list of lists of calibrated parameters
        
        assert self.n_workers == 1 or self.n_runs == 1, f'Since daemons cannot spawn, you cannot parallelize both workers ({self.n_workers}) and sims per worker ({self.n_runs})'

        # Control plotting
        self.to_plot = to_plot



    def create_sim(self, x, verbose=0):
        ''' Create the simulation from the parameters '''

        
        if isinstance(x, dict):
            pars, pkeys = self.get_bounds() # Get parameter guesses
            x = [x[k] for k in pkeys]
        

        # Define and load the data
        self.calibration_parameters = x
        
        # Parameters
        assert len(x) == len(self.pdict), 'shape of x and pdict does not match'
        
        # First calibration consists of 5 parameters
        if len(x) == 5:
            pop_infected = x[0]
            beta         = x[1]
            beta_day     = x[2]
            beta_change  = x[3]
            symp_test    = x[4]

            pars = {'pop_size': self.pop_size,
                     'pop_infected': pop_infected,
                     'pop_type': 'synthpops',
                     'location': 'Chelyabinsk',
                     'start_day': self.start_day,
                     'end_day': self.end_day,
                     'location': self.location,
                     'rand_seed': 1,
                     'verbose': 0,
                     'pop_scale': 1,
                     'scaled_pop': None,
                     'rescale': False,
                     'rescale_threshold': 1,
                     'rescale_factor': 1,
                     'frac_susceptible': 0.78,
                     'contacts': {'h': 1.57, 's': 8.5, 'w': 8.5, 'c': 0, 'l': 10},
                     'dynam_layer': {'h': 0, 's': 0, 'w': 0, 'c': 0, 'l': 0},
                     'beta_layer': {'h': 3.0, 's': 0.6, 'w': 0.6, 'c': 0, 'l': 1.5},
                     'beta_dist': {'dist': 'neg_binomial',
                      'par1': 1.0,
                      'par2': 0.45,
                      'step': 0.01},
                     'viral_dist': {'frac_time': 0.3, 'load_ratio': 2, 'high_cap': 4},
                     'beta': beta,
                     'asymp_factor': 1.0,
                     'n_imports': 0,
                     'n_variants': 1,
                     'use_waning': False,

                     'rel_imm_symp': {'asymp': 0.85, 'mild': 1, 'severe': 1.5},
                     'immunity': None,
                     'trans_redux': 0.59,
                     'rel_beta': 1.0,
                     'interventions': [cv.test_num(daily_tests=40797, symp_test=100.0, quar_test=1.0, quar_policy=None, subtarget=None, 
                                                   ili_prev=None, sensitivity=1.0, loss_prob=0, test_delay=0, start_day=0, end_day=None, 
                                                   swab_delay=None)],
                     'rel_symp_prob': 1.0,
                     'rel_severe_prob': 1.0,
                     'rel_crit_prob': 0,
                     'rel_death_prob': 0,
                     'prog_by_age': False,
                     'prognoses': {'age_cutoffs': np.array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]),
                          'sus_ORs':  np.array([0.34, 0.67, 1.  , 1.  , 1.  , 1.  , 1.24, 1.47, 1.47, 1.47]),
                          'trans_ORs':  np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]),
                          'comorbidities':  np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]),
                          'symp_probs':  np.array([0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.9 ]),
                          'severe_probs':  np.array([0.001, 0.003, 0.012, 0.032, 0.049, 0.102, 0.166, 0.243, 0.273,
                                 0.273]),
                          'crit_probs':  np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                          'death_probs':  np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])},
                     'iso_factor': {'h': 1, 's': 1, 'w': 1, 'c': 1, 'l': 1},
                     'quar_factor': {'h': 1, 's': 1, 'w': 1, 'c': 1, 'l': 1},
                     'quar_period': 0,
                     'analyzers': [],
                     'timelimit': None,
                     'stopping_func': None,
                     'n_beds_hosp': None,
                     'n_beds_icu': None,
                     'no_hosp_factor': 1,
                     'no_icu_factor': 1,
                     'vaccine_pars': {},
                     'vaccine_map': {},
                     'variants': [],
                     'variant_map': {0: 'wild'},
                     'variant_pars': {'wild': {'rel_beta': 1.0,
                       'rel_symp_prob': 1.0,
                       'rel_severe_prob': 1.0,
                       'rel_crit_prob': 0,
                       'rel_death_prob': 0}}}

            # Create the sim
            sim = cv.Sim(pars, datafile=self.datafile)
            sim.popdict = cv.make_synthpop(sim, popdict=self.popdict)
            interventions = [cv.change_beta(days=beta_day, changes=beta_change), # different beta_changes
                        cv.test_num(daily_tests=sim.data['new_tests'].dropna(), symp_test=symp_test) # testing routine
                        ]
            if self.school_days is not None and self.school_changes is not None:
                interventions+=[cv.clip_edges(days=self.school_days, changes=self.school_changes, layers='s')] # schools closed 
            
        # Other calibrations consist of 2 parameters    
        if len(x) == 2:
            beta_day    = x[0]
            beta_change = x[1]
            symp_test=self.cal[0][4]
            
            pars = {'pop_size': self.pop_size,
                     'pop_infected': self.cal[0][0],
                     'pop_type': 'synthpops',
                     'location': 'Chelyabinsk',
                     'start_day': self.start_day,
                     'end_day': self.end_day,
                     'location': self.location,
                     'rand_seed': 1,
                     'verbose': 0,
                     'pop_scale': 1,
                     'scaled_pop': None,
                     'rescale': False,
                     'rescale_threshold': 1,
                     'rescale_factor': 1,
                     'frac_susceptible': 0.78,
                     'contacts': {'h': 1.57, 's': 8.5, 'w': 8.5, 'c': 0, 'l': 10},
                     'dynam_layer': {'h': 0, 's': 0, 'w': 0, 'c': 0, 'l': 0},
                     'beta_layer': {'h': 3.0, 's': 0.6, 'w': 0.6, 'c': 0, 'l': 1.5},
                     'beta_dist': {'dist': 'neg_binomial',
                      'par1': 1.0,
                      'par2': 0.45,
                      'step': 0.01},
                     'viral_dist': {'frac_time': 0.3, 'load_ratio': 2, 'high_cap': 4},
                     'beta': self.cal[0][1],
                     'asymp_factor': 1.0,
                     'n_imports': 0,
                     'n_variants': 1,
                     'use_waning': False,

                     'rel_imm_symp': {'asymp': 0.85, 'mild': 1, 'severe': 1.5},
                     'immunity': None,
                     'trans_redux': 0.59,
                     'rel_beta': 1.0,
                     'interventions': [cv.test_num(daily_tests=40797, symp_test=100.0, quar_test=1.0, quar_policy=None, subtarget=None, 
                                                   ili_prev=None, sensitivity=1.0, loss_prob=0, test_delay=0, start_day=0, end_day=None, 
                                                   swab_delay=None)],
                     'rel_symp_prob': 1.0,
                     'rel_severe_prob': 1.0,
                     'rel_crit_prob': 0,
                     'rel_death_prob': 0,
                     'prog_by_age': False,
                     'prognoses': {'age_cutoffs': np.array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]),
                          'sus_ORs':  np.array([0.34, 0.67, 1.  , 1.  , 1.  , 1.  , 1.24, 1.47, 1.47, 1.47]),
                          'trans_ORs':  np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]),
                          'comorbidities':  np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]),
                          'symp_probs':  np.array([0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.9 ]),
                          'severe_probs':  np.array([0.001, 0.003, 0.012, 0.032, 0.049, 0.102, 0.166, 0.243, 0.273,
                                 0.273]),
                          'crit_probs':  np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                          'death_probs':  np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])},
                     'iso_factor': {'h': 1, 's': 1, 'w': 1, 'c': 1, 'l': 1},
                     'quar_factor': {'h': 1, 's': 1, 'w': 1, 'c': 1, 'l': 1},
                     'quar_period': 0,
                     'analyzers': [],
                     'timelimit': None,
                     'stopping_func': None,
                     'n_beds_hosp': None,
                     'n_beds_icu': None,
                     'no_hosp_factor': 1,
                     'no_icu_factor': 1,
                     'vaccine_pars': {},
                     'vaccine_map': {},
                     'variants': [],
                     'variant_map': {0: 'wild'},
                     'variant_pars': {'wild': {'rel_beta': 1.0,
                       'rel_symp_prob': 1.0,
                       'rel_severe_prob': 1.0,
                       'rel_crit_prob': 0,
                       'rel_death_prob': 0}}}

            # Create the sim
            sim = cv.Sim(pars, datafile=self.datafile)
            sim.popdict = cv.make_synthpop(sim, popdict=self.popdict)
            # Add interventions
            interventions = [cv.change_beta(days=self.cal[i][2], changes=self.cal[i][3]) 
                              for i in range(len(self.cal))]
            interventions += [cv.change_beta(days=beta_day, changes=beta_change)]
            interventions += [cv.test_num(daily_tests=sim.data['new_tests'].dropna(), symp_test=symp_test)]
            
            if self.school_days is not None and self.school_changes is not None:
                interventions+=[cv.clip_edges(days=self.school_days, changes=self.school_changes, layers='s')] # schools closed
        # Update
        sim.update_pars(interventions=interventions)

        self.sim = sim

        return sim


    def get_bounds(self):
        ''' Set parameter starting points and bounds -- NB, only lower and upper bounds used for fitting '''

        # Convert from dicts to arrays
        pars = sc.objdict()
        for key in ['best', 'lb', 'ub']:
            pars[key] = np.array([v[key] for v in self.pdict.values()])

        return pars, self.pdict.keys()


    def smooth(self, y, sigma=3):
        ''' Optional smoothing if using daily death data '''
        return sp.ndimage.gaussian_filter1d(y, sigma=sigma)


    
    def run_msim(self):
        if self.n_runs == 1:
            sim = self.sim
            sim.run()
        else:
            msim = cv.MultiSim(base_sim=self.sim)
            msim.run(n_runs=self.n_runs)
            sim = msim.reduce(output=True)
            
        weights={self.cal_keys[i] : self.cal_values[i] for i in range(len(self.cal_keys))}
        fit = sim.compute_fit(keys=self.cal_keys, weights=weights, output=False)
        self.sim = sim
        self.mismatch = fit.mismatch
         
        return sim

    # Functions for Optuna
    def objective(self, x):
        ''' Define the objective function we are trying to minimize '''
        self.create_sim(x=x)
        self.run_msim()
        return self.mismatch


    def op_objective(self, trial):
        ''' Define the objective for Optuna '''
        pars, pkeys = self.get_bounds() # Get parameter guesses
        x = np.zeros(len(pkeys))
        for k,key in enumerate(pkeys):
            x[k] = trial.suggest_uniform(key, pars.lb[k], pars.ub[k])

        return self.objective(x)

    def worker(self):
        ''' Run a single Optuna worker '''
        study = op.load_study(storage=self.storage, study_name=self.name)
        return study.optimize(self.op_objective, n_trials=self.n_trials)


    def run_workers(self):
        ''' Run allworkers -- parallelized if each sim is not parallelized '''
        if self.n_workers == 1:
            self.worker()
        else:
            sc.parallelize(self.worker, self.n_workers)
        return


    def make_study(self):
        try: op.delete_study(storage=self.storage, study_name=self.name)
        except: pass
        return op.create_study(storage=self.storage, study_name=self.name)


    def load_study(self):
        return op.load_study(storage=self.storage, study_name=self.name)


    def get_best_pars(self, print_mismatch=True):
        ''' Get the outcomes of a calibration '''
        study = self.load_study()
        output = study.best_params
        if print_mismatch:
            print(f'Mismatch: {study.best_value}')
        return output


    def calibrate(self):
        ''' Perform the calibration '''
        self.make_study()
        self.run_workers()
        output = self.get_best_pars()
        return output


    def save(self):
        pars_calib = self.get_best_pars()
        sc.savejson(f'calibrated_parameters_{self.until}_{self.state}.json', pars_calib)


# Modelling after calibration
        
def model(cal):
    
    recalibrate = True # Whether to run the calibration
    do_plot     = True # Whether to plot results
    verbose     = 0.1 # How much detail to print

   

    # Plot initial
    if do_plot:
        print('Running initial uncalibrated simulation...')
        pars, pkeys = cal.get_bounds() # Get parameter guesses
        sim = cal.create_sim(pars.best, verbose=verbose)
        sim.run()
        sim.plot(to_plot=cal.to_plot)
        pl.gcf().suptitle('Initial parameter values')
        cal.objective(pars.best)
        pl.pause(1.0) # Ensure it has time to render

    # Calibrate
    if recalibrate:
        print(f'Starting calibration for Novosibirsk')
        T = sc.tic()
        pars_calib = cal.calibrate()
        sc.toc(T)
    else:
        pars_calib = cal.get_best_pars()

    # Plot result
    if do_plot:
        print('Plotting result...')
        x = [pars_calib[k] for k in pkeys]
        sim = cal.create_sim(x, verbose=verbose)
        sim.run()
        sim.plot(to_plot=cal.to_plot)
        pl.gcf().suptitle('Calibrated parameter values')
    return sim, pars_calib   



