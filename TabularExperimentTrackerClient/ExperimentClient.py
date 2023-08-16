import openml
import requests
import json

class ExperimentClient:
    """For managing communication with orchestrator and loading data

    the idea is for this to be a pip-installable module hosted on
    github.
    """

    def __init__(self, verbose=False, opml_regression=True, opml_classification=True, opml_purnum = True, opml_numcat = True, runs_per_pair=60):
        """Initialize
        verbose: helpful printouts or not
        opml_regression: tasks regression targets
        opml_classification: tasks classification targets
        opml_purnum: tasks with pure numeric features
        opml_numcat: tasks with numeric and categorical features
        """

        #no experiment defined yet
        self.expname = None
        self.model_groups = None
        self.data_groups = None
        self.applications = None

        #no run started yet
        self.run_id = None

        #no secerets defined yet
        self.orchname = None
        self.orchseceret = None
        self.openMLAPIKey = None

        #user defined verbosity
        self.verbose=verbose

        #defines which suites to use. classification, regression, pure numeric, and numeric/categorical
        self.suites_ids = []
        if opml_regression:
            if opml_purnum:
                self.suites_ids.append(336)
            if opml_numcat:
                self.suites_ids.append(335)
        if opml_classification:
            if opml_purnum:
                self.suites_ids.append(337)
            if opml_numcat:
                self.suites_ids.append(334)

        #get queried when openMLAPIKey is defined
        self.suites = []
        self.taskID_suite = []

        #how many times to run a hyperparameter-task pair
        self.runs_per_pair = runs_per_pair

    def mount_drive(self):
        """ Mounts a drive, if on google colab, chiefly for getting credentials
        Works if working on google colab. *shouldn't* break anything if
        not called and not on colab
        """
        from google.colab import drive
        drive.mount('/content/drive')

    #===========================================================================
    #                      Auth and Seceret Management
    #
    # There are two main auth flows: explicitely defined strings, or loading
    # credentials from a location on google drive (for colab)
    #===========================================================================

    def define_orch_cred(self, orchname, orchseceret):
        """Define all the credentialls necessary for communicating with the orchestrator
        orchname: passes to the name header
        orchseceret: passes to the seceret header
        """
        self.orchname = orchname
        self.orchseceret = orchseceret

    def define_orch_cred_drive(self, orchname, path):
        """Reads the orchestrator seceret from some drive location
        path: the path to the orch seceret, should be a .txt, as defined in
        drive itself.
        """
        self.orchname = orchname
        self.mount_drive()
        with open ('/content/drive'+path, "r") as myfile:
            self.orchseceret = myfile.read()

    def define_opml_cred(self, openMLAPIKey):
        self.openMLAPIKey = openMLAPIKey
        self.init_opml()

    def define_opml_cred_drive(self, path):
        """Reads the API key for openml from some drive location
        path: the path to the api key, should be a .txt, as defined in
        drive itself.
        """
        #setting up API key
        self.mount_drive()
        with open ('/content/drive'+path, "r") as myfile:
            self.openMLAPIKey = myfile.read()
        openml.config.apikey = self.openMLAPIKey
        self.init_opml()

    #===========================================================================
    #                           Internal Setup
    #
    # Things necessary for setting up an experiment
    #===========================================================================

    def init_opml(self):
        """Loads suite and task information
        this automatically gets called when opml credentials are set up
        """
        self.suites = [openml.study.get_suite(id) for id in self.suites_ids]
        self.taskID_suite = [(task_id, suite) for suite in self.suites for task_id in suite.tasks]

    def opml_identifiers(self):
        """Turns tasks into string which can be sent to orchestrator
        the individual identifiers are "<suiteid>-<taskid>", these are then
        packaged into a dictionary, grouping the data into groups. This can be
        used as the data groups in the experiment definition.

        data_groups: dictionary of <suite>-<task> ids, grouped
        """
        task_ids = ['{}-{}'.format(suite.study_id, task) for (task, suite) in self.taskID_suite]
        data_groups = {'opml_reg_purnum_group': [t for t in task_ids if int(t[:3]) == 336],
                       'opml_class_purnum_group': [t for t in task_ids if int(t[:3]) == 337],
                       'opml_reg_numcat_group': [t for t in task_ids if int(t[:3]) == 335],
                       'opml_class_numcat_group': [t for t in task_ids if int(t[:3]) == 334]}
        return data_groups


    #===========================================================================
    #                          Experiment Definition
    #
    # Things defined by the user to setup an experiment. A lot of rubber meets
    # the road here, so I have a lot of data integrity tests
    #===========================================================================
    def def_model_groups(self, model_groups):
        """Define experiment Model Groups
        a "model group" is a grouping of model and hyperparameter space. The 
        "model_groups" dictionary should look like:

        {<model group id>: {model:<model id>, hype:<hyperparameters>}}

        often the group id and the model id are identical, but don't necissarily
        have to be.

        The hyperparameter dictionary is a key value pair, where the value is
        yet another dictionary specifying the distribution for that particular
        hyperparameter, the name of which is the key.

        as an example, a valid hyperparameter space might look like:

        hype = {"hype1":{"distribution":"constant", "value":2.71828},
            "hype2":{"distribution":"categorical", "values":["foo", "bar"]},
            "hype3":{"distribution":"int_uniform", "min":0, "max":2},
            "hype4":{"distribution":"float_uniform", "min":0.0, "max":2.5},
            "hype5":{"distribution":"log_uniform", "min":0.0, "max":2.5}}

        this `hype` dictionary is the value for the `hype` key in a model group

        the hyperparameter space is not checked. If a hyperparameter isn't valid,
        it won't be properly searched via random search.
        """

        #ensuring integrity
        for k, v in model_groups.items():
            #checking general content
            if type(v) is not dict:
                raise(Exception("model group '{}' does not have a coresponding dictionary".format(k)))
            if set(['model', 'hype']) != set(v.keys()):
                raise(Exception("model group '{}' does not have the correct attributes of 'model' and 'hype', exclusively".format(k)))
            if type(v['model']) is not str:
                raise(Exception("model group '{}' has a non-string model (aka model id) attribute".format(k)))
            if type(v['hype']) is not dict:
                raise(Exception("model group '{}' has a non-dictionary hype (aka hyperparameter space) attribute".format(k)))
            
        self.model_groups=model_groups

    def def_data_groups_opml(self):
        """Defines data groups automatically from openml
        """
        self.data_groups = self.opml_identifiers()

    def def_data_groups(self):
        """Allows for custom defined data groups
        """
        raise(Exception("function not yet defined"))

    def def_applications(self, applications):
        """Define applications
        applications are where models are applied to data groups.
        it's a dictionary of lists, consisting of:

        {<data group>: <list of model groups>}

        in order to aid in usability, I'm checking for issues with
        the application definition. As a result, it's required
        to define data and model groups first
        """

        #checking integrity
        for k, v in applications.items():

            if k not in self.data_groups.keys():
                raise(Exception("data group '{}' in application not in '{}', which are the data groups specified".format(k, self.data_groups.keys())))

            if not isinstance(v, list):
                raise(Exception("at data group '{}', '{}' should be a list of model ids, not a {}".format(k, v, type(v))))
            
            for mid in v:
                if type(mid) != str:
                    raise(Exception("model id '{}', in '{}', should be of type 'str', not {}".format(mid, k, type(mid))))

                if mid not in self.model_groups.keys():
                    raise(Exception("model id '{}', in '{}', could not be found in defined model groups: {}".format(mid, k, self.model_groups.keys())))

        self.applications = applications

    def reg_experiment(self, expname):
        self.expname = expname

        if type(expname) != str:
            raise(Exception("expname should be a string"))

        if self.model_groups is None:
            raise(Exception("model_groups not defined, use 'def_model_groups'"))
        if self.data_groups is None:
            raise(Exception("data_groups not defined, use 'def_data_groups_opml'"))
        if self.applications is None:
            raise(Exception("applications not defined, use 'def_applications'"))

        experiment_definition = {
            "name": self.expname,
            "runs_per_pair": self.runs_per_pair,
            "definition": {
                "data_groups": self.data_groups,
                "model_groups": self.model_groups,
                "applications": self.applications
            }
        }

        url = "https://us-west-2.aws.data.mongodb-api.com/app/experimentmanager-sjmvq/endpoint/registerExperiment"
        payload = json.dumps(experiment_definition)
        headers = {'Name': self.orchname,'Seceret': self.orchseceret,'Content-Type': 'application/json'}
        response = requests.request("POST", url, headers=headers, data=payload)
        if self.verbose:
            print(response.text)
        return response.text

    #===========================================================================
    #                          Running Experiment
    #
    # Communication which begins a run, gets run info, updates a run,
    # marks a run as complete
    #===========================================================================
    def begin_run(self):
        url = "https://us-west-2.aws.data.mongodb-api.com/app/experimentmanager-sjmvq/endpoint/beginRun"
        payload = json.dumps({'experiment': self.expname})
        headers = {'Name': self.orchname,'Seceret': self.orchseceret,'Content-Type': 'application/json'}
        self.run_id = requests.request("POST", url, headers=headers, data=payload).text[1:-1] #trimming of quotes
        if self.verbose:
            print('new run id: {}'.format(self.run_id))
        return self.run_id

    def get_run(self):
        url = "https://us-west-2.aws.data.mongodb-api.com/app/experimentmanager-sjmvq/endpoint/getRun"
        payload = json.dumps({"run": self.run_id})
        headers = {'Name': self.orchname,'Seceret': self.orchseceret,'Content-Type': 'application/json'}
        currun = json.loads(requests.request("POST", url, headers=headers, data=payload).text)
        if self.verbose:
            print(currun)
        return currun

    def update_run(self, metrics):
        url = "https://us-west-2.aws.data.mongodb-api.com/app/experimentmanager-sjmvq/endpoint/updateRun"
        payload = json.dumps({"run": self.run_id,"metrics": metrics})
        headers = {'Name': self.orchname,'Seceret': self.orchseceret,'Content-Type': 'application/json'}
        resp = requests.request("POST", url, headers=headers, data=payload).text
        if self.verbose:
            print(resp)
        return resp

    def end_run(self):
        url = "https://us-west-2.aws.data.mongodb-api.com/app/experimentmanager-sjmvq/endpoint/getRun"
        payload = json.dumps({"run": self.run_id})
        headers = {'Name': self.orchname,'Seceret': self.orchseceret,'Content-Type': 'application/json'}
        currun = json.loads(requests.request("POST", url, headers=headers, data=payload).text)
        if self.verbose:
            print('completed run:')
            print(currun)
        return currun