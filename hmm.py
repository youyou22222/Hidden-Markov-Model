#!/usr/bin/env python


identity = lambda x: x

class HiddenMarkovModel(object):
    """A hidden Markov model."""
    
    def __init__(self, states, transitions, emissions, vocab):
        """
        states - a list/tuple of states, e.g. ('start', 'hot', 'cold', 'end')
                 start state needs to be first, end state last
                 states are numbered by their order here
        transitions - the probabilities to go from one state to another
                      transitions[from_state][to_state] = prob
        emissions - the probabilities of an observation for a given state
                    emissions[state][observation] = prob
        vocab: a list/tuple of the names of observable values, in order
        """
        self.states = states
        self.real_states = states[1:-1]
        self.start_state = 0
        self.end_state = len(states) - 1
        self.transitions = transitions
        self.emissions = emissions
        self.vocab = vocab
    
    # functions to get stuff one-indexed
    state_num = lambda self, n: self.states[n]
    state_nums = lambda self: xrange(1, len(self.real_states) + 1)
    
    vocab_num = lambda self, n: self.vocab[n - 1]
    vocab_nums = lambda self: xrange(1, len(self.vocab) + 1)
    num_for_vocab = lambda self, s: self.vocab.index(s) + 1
    
    def transition(self, from_state, to_state):
        return self.transitions[from_state][to_state]
    
    def emission(self, state, observed):
        return self.emissions[state][observed - 1]
    
    
    # helper stuff
    def _normalize_observations(self, observations):
        return [None] + [self.num_for_vocab(o) if o.__class__ == str else o
                                               for o in observations]
    
    def _init_trellis(self, observed, forward=True, init_func=identity):
        trellis = [ [None for j in range(len(observed))]
                          for i in range(len(self.real_states) + 1) ]
        
        if forward:
            v = lambda s: self.transition(0, s) * self.emission(s, observed[1])
        else:
            v = lambda s: self.transition(s, self.end_state)
        init_pos = 1 if forward else -1
        
        for state in self.state_nums():
            trellis[state][init_pos] = init_func( v(state) )
        return trellis
    
    def _follow_backpointers(self, trellis, start):
        # don't bother branching
        pointer = start[0]
        seq = [pointer, self.end_state]
        for t in reversed(xrange(1, len(trellis[1]))):
            val, backs = trellis[pointer][t]
            pointer = backs[0]
            seq.insert(0, pointer)
        return seq
    
    
    # actual algorithms
    
    def forward_prob(self, observations, return_trellis=False):
        """
        Returns the probability of seeing the given `observations` sequence,
        using the Forward algorithm.
        """
        observed = self._normalize_observations(observations)
        trellis = self._init_trellis(observed)
        
        for t in range(2, len(observed)):
            for state in self.state_nums():
                trellis[state][t] = sum(
                    self.transition(old_state, state)
                        * self.emission(state, observed[t])
                        * trellis[old_state][t-1]
                    for old_state in self.state_nums()
                )
        final = sum(trellis[state][-1] * self.transition(state, -1)
                    for state in self.state_nums())
        return (final, trellis) if return_trellis else final
    
    
    def backward_prob(self, observations, return_trellis=False):
        """
        Returns the probability of seeing the given `observations` sequence,
        using the Backward algorithm.
        """
        observed = self._normalize_observations(observations)
        trellis = self._init_trellis(observed, forward=False)
        
        for t in reversed(range(1, len(observed) - 1)):
            for state in self.state_nums():
                trellis[state][t] = sum(
                    self.transition(state, next_state)
                        * self.emission(next_state, observed[t+1])
                        * trellis[next_state][t+1]
                    for next_state in self.state_nums()
                )
        final = sum(self.transition(0, state)
                        * self.emission(state, observed[1])
                        * trellis[state][1]
                    for state in self.state_nums())
        return (final, trellis) if return_trellis else final
    
    
    def viterbi_sequence(self, observations, return_trellis=False):
        """
        Returns the most likely sequence of hidden states, for a given
        sequence of observations. Uses the Viterbi algorithm.
        """
        observed = self._normalize_observations(observations)
        trellis = self._init_trellis(observed, init_func=lambda val: (val, [0]))
        
        for t in range(2, len(observed)):
            for state in self.state_nums():
                emission_prob = self.emission(state, observed[t])
                last = [(old_state, trellis[old_state][t-1][0] * \
                                    self.transition(old_state, state) * \
                                    emission_prob)
                        for old_state in self.state_nums()]
                highest = max(last, key=lambda p: p[1])[1]
                backs = [s for s, val in last if val == highest]
                trellis[state][t] = (highest, backs)
        
        last = [(old_state, trellis[old_state][-1][0] * \
                            self.transition(old_state, self.end_state)) 
                for old_state in self.state_nums()]
        highest = max(last, key = lambda p: p[1])[1]
        backs = [s for s, val in last if val == highest]
        seq = self._follow_backpointers(trellis, backs)
        
        return (seq, trellis) if return_trellis else seq
    
    
    def train_on_obs(self, observations, return_probs=False):
        """
        Trains the model once, using the forward-backward algorithm. This
        function returns a new HMM instance rather than modifying this one.
        """
        observed = self._normalize_observations(observations)
        forward_prob,  forwards  = self.forward_prob( observations, True)
        backward_prob, backwards = self.backward_prob(observations, True)
        
        # gamma values
        prob_of_state_at_time = posat = [None] + [
            [0] + [forwards[state][t] * backwards[state][t] / forward_prob
                for t in range(1, len(observations)+1)]
            for state in self.state_nums()]
        # xi values
        prob_of_transition = pot = [None] + [
            [None] + [
                [0] + [forwards[state1][t] 
                        * self.transition(state1, state2)
                        * self.emission(state2, observed[t+1]) 
                        * backwards[state2][t+1]
                        / forward_prob
                  for t in range(1, len(observations))]
              for state2 in self.state_nums()]
          for state1 in self.state_nums()]
        
        # new transition probabilities
        trans = [[0 for j in range(len(self.states))]
                    for i in range(len(self.states))]
        trans[self.end_state][self.end_state] = 1
        
        for state in self.state_nums():
            state_prob = sum(posat[state])
            trans[0][state] = posat[state][1]
            trans[state][-1] = posat[state][-1] / state_prob
            for oth in self.state_nums():
                trans[state][oth] = sum(pot[state][oth]) / state_prob
        
        # new emission probabilities
        emit = [[0 for j in range(len(self.vocab))]
                   for i in range(len(self.states))]
        for state in self.state_nums():
            for output in range(1, len(self.vocab) + 1):
                n = sum(posat[state][t] for t in range(1, len(observations)+1)
                                              if observed[t] == output)
                emit[state][output-1] = n / sum(posat[state])
        
        trained = HiddenMarkovModel(self.states, trans, emit, self.vocab)
        return (trained, posat, pot) if return_probs else trained
    

# ======================
# = reading from files =
# ======================

def normalize(string):
    if '#' in string:
        string = string[:string.index('#')]
    return string.strip()

def make_hmm_from_file(f):
    def nextline():
        line = f.readline()
        if line == '': # EOF
            return None
        else:
            return normalize(line) or nextline()
    
    n = int(nextline())
    states = [nextline() for i in range(n)] # <3 list comprehension abuse
    
    num_vocab = int(nextline())
    vocab = [nextline() for i in range(num_vocab)]
    
    transitions = [[float(x) for x in nextline().split()] for i in range(n)]
    emissions   = [[float(x) for x in nextline().split()] for i in range(n)]
    
    assert nextline() is None
    return HiddenMarkovModel(states, transitions, emissions, vocab)

def read_observations_from_file(f):
    return filter(lambda x: x, [normalize(line) for line in f.readlines()])

