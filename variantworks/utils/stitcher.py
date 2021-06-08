#
# Copyright 2020 NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Stitcher Utilities.

Combine chunk predictions into a sequence.
"""

from collections import namedtuple
import numpy as np
import sys

NucleotideCertainty = namedtuple('NucleotideCertainty', ['nucleotide_literal', 'nucleotide_certainty'])


def decode_consensus(probs, include_certainty_score=True):
    """Decode probabilities into sequence by choosing the Nucleotide base with the highest probability.

    Returns:
        seq: sequence output from probabilities
    """

    #### determine output type from length 25=outputtype=="matchins25" 5=outputtype=="singlebase"
    #### TODO: DANGEROUS implicit length == output type !!!
    outputtype="null"
    mylen=len(probs[0,:])
    if mylen==25:
        outputtype="matchins25"
    elif mylen==5:
        outputtype="singlebase"

    # todo: doesn't work!
    # only print on first invocation to cut down on output volume. Uses attribute on function
    if not hasattr('decode_consensus', 'called'):
        #print("decode_consensus outputtype",outputtype)
        setattr(decode_consensus, "called", "True")

    if outputtype=="matchins25":
        # label_sybols are now 0:24 and represent match+insertFollow
        # base2ind = {"A": 0, "a": 0, "C": 1, "c": 1, "G": 2, "g": 2, "T": 3, "t": 3, "-": 4}
        # self.labelsTensor[ww,ii] = base2ind[matchbase]+5*base2ind[insertbase[0]]; 
        # 0 = A+5*insA
        # 1 = C+5*insA
        # 24 = "" = -+5ins-
        label_symbols = []
        for insert in ["a","c","g","t",""]:
            for match in ["A","C","G","T",""]:
                label_symbols.append("%s%s" % (match,insert))
    elif outputtype=="singlebase":
        # single base
        label_symbols = ["", "A", "C", "G", "T"]  # Corresponding labels for each network output channel
    else:
        print("ERROR: decode_consensus outputtype not understood", outputtype)
        sys.exit(1)

    seq = []
    seq_quality = []
    # take max along softmax axis
    mp = np.argmax(probs,1)
    for ii in range(len(probs)):
        nuc = label_symbols[mp[ii].item()]
        if nuc != "":
            seq.append(nuc)
            if include_certainty_score:
                myprob = probs[ii,mp[ii]].item()
                seq_quality.append(myprob)
                if len(nuc)==2: seq_quality.append(myprob) # match+insert give same qv for matchins25
    seq="".join(seq)
    return seq, seq_quality if include_certainty_score else list()

def posFind( key, data, hint=0 ):
    """find key in data where data is sorted. data is (pos,0). short
    linear search but keeps from running recursive search and getting
    edge cases wrong

    """

    index = hint
    here = data[index][0]
    if key[0] == here:
        return(index)

    if key[0] > here:
        # go right
        for index in range(hint+1,len(data),1):
            here = data[index][0]
            if key[0] == here:
                return(index)
        sys.exit(1)
        
    if key[0] < here:
        # go left
        for index in range(hint-1,-1,-1):
            here = data[index][0]
            if key[0] == here:
                return(index)
        sys.exit(1)
        
def stitch(probs, positions, decode_consensus_func):
    """Stitch predictions on chunks into a contiguous sequence.

    Args:
        probs: 3D array of predicted probabilities. no. of chunks X  no. of positions in chunk X no. of bases.
        positions: Corresponding list of position array for each chunk in probs.
        decode_consensus_func: A function which decodes each chunk from probs into label_symbols.

    The current code appears to be much more complex than necessary!

    Chunks are fixed window sizes with overlap. We trust the ends a
    bit less because of reuced context. Note this is only one
    way. Could take half.

    [ Chunk0 positions ]                              
    [**************)
                ss0|   |ee1                        
                   [ Chunk1 positions ]               
                   [**************)
                               ss1|   |ee2         
                                  [ Chunk2 positions ]
                                  [**************)
                                              ss1|   |ee2         
                                                 [ Chunk2 positions ]
                                                 [******************]

    ss = next_start_in_this
    ee = this_end_in_next

    Returns:
        seq: Stitched consensus sequence

    """
    windowHint=31
    windowLen=len(positions[0])
                  
    decoded_sequece_parts = list()

    #### handle first
    ii=0
    first_start_idx=0
    ss=posFind(positions[ii+1][0], positions[ii], max(0,windowLen-windowHint))
    ee=posFind(positions[ii ][-1], positions[ii+1],min(windowLen,windowHint))
    half = int(ee/2+0.5)
    if len(positions)>1:
        first_end_idx=ss+half
    else:
        first_end_idx=len(positions[ii])
    second_start_idx=0+half
    #print("first_start_idx, first_end_idx, second_start_id",first_start_idx, first_end_idx, second_start_idx)
    decoded_seq = decode_consensus_func(probs[ii][first_start_idx:first_end_idx])
    decoded_sequece_parts.append(decoded_seq)

    #### handle interior
    for ii in range(1, len(positions)):
        first_start_idx=second_start_idx
        if ii < len(positions)-1:
            ss=posFind(positions[ii+1][0], positions[ii], max(0,windowLen-windowHint))
            ee=posFind(positions[ii ][-1], positions[ii+1],min(windowLen,windowHint))
            half = int(ee/2+0.5)
            first_end_idx=ss+half
        else:
            first_end_idx=len(positions[ii])
        second_start_idx=0+half
        #print("first_start_idx, first_end_idx, second_start_id",first_start_idx, first_end_idx, second_start_idx)
        decoded_seq = decode_consensus_func(probs[ii][first_start_idx:first_end_idx])
        decoded_sequece_parts.append(decoded_seq)

    return decoded_sequece_parts
