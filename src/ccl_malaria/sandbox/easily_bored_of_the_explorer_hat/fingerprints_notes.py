# coding=utf-8
"""
Fingerprinting playground.

- Dalke's chemfp: http://code.google.com/p/chem-fingerprints/
                  http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3083566/
                  http://chemfp.com/
- Galaxy tools (look for chemfp): https://github.com/bgruening/galaxytools
- RDKit PYAvalonTools: http://www.rdkit.org/docs/api/rdkit.Avalon.pyAvalonTools-module.html
- Revisiting molecular hashing fingerprints: http://chembioinfo.com/2011/10/30/revisiting-molecular-hashed-fingerprints/
- Cheminformatics toolkits: http://en.wikipedia.org/wiki/Cheminformatics_toolkits
- Many others, something interesting under the sun?

We need to support:
  - A map from molecule to fingerprint counts
  - A unique map from smiles to integer and back.
  - Is it useful to store for each fingerprint bit the center and radius?

Several drawbacks should be noted as opposed to using a hash function:
  - The s -> i depends on the order the molecules are presented to the gatherer
  - Adding incrementally "bits" to consider is not so straightforward
So this approach is most useful for static collections of molecules.


There are several possible ways of supporting fingerprint generation:
- Batch: input all molecules that will generate features, create the dictionaries maybe sorting by size
- Online: ...

TODO look at nice monkey-patching here:
from rdkit.Chem import PandasTools

TODO: probably it is best to sort features by frequency of appearance when recoding
TODO: Do not use bond type (useBondTypes=False)
TODO: Use own invariants (what is an invariant?)
"""