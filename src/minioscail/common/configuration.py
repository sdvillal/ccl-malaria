# coding=utf-8
"""An attempt to abstract configurability and experiment identification."""
import datetime
import hashlib
import inspect
from copy import copy
from collections import OrderedDict
from functools import partial
from socket import gethostname
from minioscail.common.misc import partial2call, is_iterable

###########################################
# Info helpers
###########################################
# Ways of pretty printing info
# -Triple quotes + dedent
#
# info = dedent(u"""
#              Randomly-setup scikit random-forest.
#              Date=%s
#              Data-info=%s
#              SetupString=%s
#              """%(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), data.id(), name))
#
# - Json
# info = json.dumps({
#    'title' : 'Randomly-setup scikit random-forest',
#    'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
#    'data-info' : data.id(),
#    'setup-string' : name
# }, indent=2)
###########################################
# TODOs:
#   - Normalize time with some internet service (problem: computers without access, like cluster nodes). Best effort.
#   - Current function name and source
#   - Git revision + is_dirty.
#     We could agree on always committing to a particular branch, but I feel that is overkill ATM.
#   - Implementation, libraries version etc could be candidates for writting too
###########################################


def mlexp_info_helper(title,
                      data_setup=None,
                      model_setup=None,
                      exp_function=None,
                      comments=None):
    """Creates an information dictionary describing or summarizing machine learning experiments.

    Parameters:
      - title: the title for the experiment, maybe a short identifier
      - data_setup: None or a configurable (or an object providing an "id" method) for the data used in the experiment
      - model_setup: None or a configurable (or an object providing an "id" method) for the model used in the experiment
      - exp_function: the function in which the experiment is defined
      - comments: a string with whatever else we need to say

    Return:
      An orderec dict
    """

    def setup_as_string(obj):
        if obj is None:
            return None
        if isinstance(obj, basestring):
            return obj
        try:
            return obj.id()
        except AttributeError:
            raise TypeError('the object must be None, a string or have an id() method')  # Lame

    info = OrderedDict((
        ('title', title),
        ('data_setup', setup_as_string(data_setup)),
        ('model_setup', setup_as_string(model_setup)),
        ('fsource', inspect.getsourcelines(exp_function) if exp_function else None),
        ('comments', comments),
        ('date', datetime.datetime.now().strftime("%Y-%m-%d %H:%M")),
        ('host', gethostname()),
    ))
    return info


def function_stack_code(base_function):
    inspect.getsourcelines(base_function)


###########################################
# Configurable objects
###########################################
#
# For a nice example using introspection, see sklearn.base.BaseEstimator.
# In sklearn all parameters must be passed in the constructor, which is a fundamental limitation.
# Or can this limitation be overriden by the usage of python properties?
#
# TODO: annotate classes instead of inheritance?
#       Python and mixins:
#       - http://stackoverflow.com/questions/533631/what-is-a-mixin-and-why-are-they-useful
#       Actually duck typing should not be used here, a configuration() method may well exist for other objects.
#       Let's just use object orientation for this
#
# TODO: manage property deprecation
#       sklearn does it nicely, but it enables DeprecationWarnings regardless of whatever has happened before
#
# TODO: types of parameters
#       probably not a good idea, but give a look to things like pycontracts
#       http://andreacensi.github.io/contracts/
#
# TODO: configure and clone (ala sklearn);
#       problem: set parameters might not be as straightforward,
#                and maybe that's why they require to put them in the constructor in sklearn
#                but we can use python properties for that...
#
# TODO: parse a configuration string back
#       this is coupled to clone and configure
#       we should alse keep a factory from name to constructor... boring
#
############################################

def dict_or_slotsdict(obj):
    """Returns a dictionary with the properties for an object, regardless if it has __slots__ defined.
    Example:
    >>> class NoSlots(object):
    ...     def __init__(self):
    ...         self.prop = 3
    >>> dict_or_slotsdict(NoSlots())
    {'prop': 3}
    >>> class Slots(object):
    ...     __slots__ = ['prop']
    ...     def __init__(self):
    ...         self.prop = 3
    >>> dict_or_slotsdict(Slots())
    {'prop': 3}
    """
    try:
        return obj.__dict__
    except:
        return {slot: getattr(obj, slot) for slot in obj.__slots__}


def data_descriptors(obj):
    """
    Example:
    >>> class PropertyCarrier(object):
    ...     def __init__(self):
    ...         self._prop = 3
    ...     @property
    ...     def prop(self):
    ...         return self._prop
    ...     @prop.setter
    ...     def prop(self, prop):
    ...         self._prop = prop
    >>> data_descriptors(PropertyCarrier())
    {'prop': 3}

    """
    #
    # http://www.gossamer-threads.com/lists/python/python/929510
    #
    # Consenting Adults
    #
    descriptors = inspect.getmembers(obj.__class__, inspect.isdatadescriptor)
    descriptors = inspect.getmembers(obj)
    # But what about dynamically added descriptors?
    # It should be possible to take into account this corner case too...
    return descriptors


class Configurable(object):
    """A configurable object has a configuration."""

    def __init__(self):
        super(Configurable, self).__init__()

    def config_dict(self):
        """Returns the object instance dictionary but for a property called "configuration"."""
        return Configurable.config_dict_for_object(self)

    @staticmethod
    def config_dict_for_object(obj):
        """Returns the object instance dictionary but for a property called "configuration"."""
        import inspect
        # Data descriptors
        data_descriptors = inspect.getmembers(obj, inspect.isdatadescriptor)
        # __slots__ or __dict__
        dict_or_slotsdict(obj)

        config = {}
        for k, v in obj.__dict__.iteritems():  # What about properties? What about __slots__?
            # See: http://docs.python.org/2/reference/datamodel.html
            # http://stackoverflow.com/questions/17330160/
            # python-how-does-decorator-property-work
            # Use inspect and isdatadescriptor...
            # inspect.getmembers(obj, inspect.isdatadescriptor)
            if k != 'configuration' and not k.startswith('_') and not k.endswith('_'):
                config[k] = v
        return config

    def configuration(self):  # NEW: not a property anymore
        """Returns a Configuration object."""
        return Configuration(
            self.__class__.__name__,
            configuration_dict=self.config_dict()
        )

    def configure(self, configuration):
        """Configures the current object with the requested configuration.
        By default we assumed that each parameter corresponds to one property of self.
        """
        # TODO: use six
        # TODO: is this a good idea? creation of configurable objects should
        #       be anyway cheap...
        raise NotImplementedError


class Configuration(object):
    """
    A (sorted-dictionary-based) configurable object helper class.
    Configurations can nest and be reasonably represented by strings.
    """

    def __init__(self,
                 name,
                 configuration_dict,
                 non_ids=None,
                 display_synonyms=None,
                 kv_separator='__',
                 kvsv_separator='=',
                 nested_separator='\"\"',
                 sorted_by_key=True,
                 first_ks=None,
                 last_ks=None,
                 length_limit=255):
        super(Configuration, self).__init__()
        self.name = name
        self.dict = configuration_dict
        self.sorted_by_key = sorted_by_key
        self.kv_separator = kv_separator
        self.kvsv_separator = kvsv_separator
        self.nested_separator = nested_separator
        self.length_limit = length_limit
        self.display_synonyms = display_synonyms if display_synonyms else {}
        self.first_ks = first_ks if first_ks else []
        self.last_ks = last_ks if last_ks else []
        #Keys here won't make it to the configuration string
        if not non_ids:
            self.non_ids = set()
        elif is_iterable(non_ids):
            self.non_ids = set(non_ids)
        else:
            raise Exception('non_ids must be None or an iterable')

    def configuration_string(self, full=False):
        """Returns a string representing this configuration using our formatting parameters."""
        #Key-value list
        def sort_kvs_fl():
            kvs = self.dict.iteritems()
            if self.sorted_by_key:
                kvs = sorted(kvs)
            first_set = set(self.first_ks)
            last_set = set(self.last_ks)
            if len(first_set & last_set) > 0:
                raise Exception('Some identifiers (%r) appear in both first and last, they should not' %
                                (first_set & last_set))
            kvs_dict = dict(kvs)
            return [(f, kvs_dict[f]) for f in self.first_ks] + \
                   [kv for kv in kvs if not kv[0] in (first_set | last_set)] + \
                   [(f, kvs_dict[f]) for f in self.last_ks]

        kvs = sort_kvs_fl()
        return self.kv_separator.join(
            '%s%s%s' % (self.synonym(k), self.kvsv_separator, self._nested_string(v))
            for k, v in kvs
            if full or k not in self.non_ids)

    def __getitem__(self, item):
        return self.dict[item]

    def id(self, full=False):
        return self.synonym(self.name) + self.kv_separator + self.configuration_string(full=full)

    def _nested_string(self, v):
        if isinstance(v, Configuration):
            return self.nested_separator[0] + v.id() + self.nested_separator[1]
        if isinstance(v, Configurable):
            return self.nested_separator[0] + v.configuration().id() + self.nested_separator[1]
        if inspect.isbuiltin(v):  # Special message if we try to pass something like sorted or np.array
            raise Exception('Cannot determine the argspec of a non-python function (%s). '
                            'Please wrap it in a configurable' % v.__name__)
        if isinstance(v, partial):
            name, keywords = partial2call(v)
            config = copy(self)
            config.name = name
            config.dict = keywords
            return self.nested_separator[0] + config.id() + self.nested_separator[1]
        if inspect.isfunction(v):
            args, _, _, defaults = inspect.getargspec(v)
            defaults = [] if not defaults else defaults
            args = [] if not args else args
            params_with_defaults = dict(zip(args[-len(defaults):], defaults))
            config = copy(self)
            config.name = v.__name__
            config.dict = params_with_defaults
            return self.nested_separator[0] + config.id() + self.nested_separator[1]
        if 'object at 0x' in str(v):  # An object without proper representation, try a best effort
            config = copy(self)
            config.name = v.__class__.__name__
            config.dict = Configurable.config_dict_for_object(v)
            return self.nested_separator[0] + config.id() + self.nested_separator[1]
        return str(v)

    def id_fn(self, full=False):
        """
        Returns this object id and a filename for this configuration:
        the id itself or the sha1 hash if the id is too long.
        """
        my_id = self.id(full=full)
        if not self.length_limit or len(my_id) <= self.length_limit:
            return my_id
        return self.sha2(full=full)

    def md5(self, full=False):
        return hashlib.md5(self.id(full=full)).hexdigest()

    def sha1(self, full=False):
        return hashlib.sha1(self.id(full=full)).hexdigest()

    def sha2(self, full=False):
        return hashlib.sha256(self.id(full=full)).hexdigest()

    def add_synonym(self, name, synonym):
        """Configures the synonym for the property."""
        self.display_synonyms[name] = synonym

    def synonym(self, name):
        """Returns the global synonym for the property name, if it is registered, otherwise the name itself."""
        return self.display_synonyms.get(name, name)


def split_by(cfg, separator='__', one_char_sep='#'):
    """
    Splits a string separated by the separator, with quotes and with key=value structure.
    Expects: name__k1=v1__k2="name__k22=v22"__k3=v3.
    Read the damn code.
    """
    cfg = cfg.replace(separator, one_char_sep)  # TODO: Check that there is no a hash already present in the string
    import shlex

    splitter = shlex.shlex(instream=cfg)
    splitter.wordchars += '.'  # So numbers are not splitted
    splitter.whitespace = one_char_sep
    splitter.whitespace_split = False
    values = list(splitter)
    name = values[0]
    assert all(val == '=' for val in values[2::3]), 'Splitting has not work.' \
                                                    'There is something that is not a = where the should be.'
    return name, dict(zip(values[1::3], values[3::3]))