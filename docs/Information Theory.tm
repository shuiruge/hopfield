<TeXmacs|1.99.10>

<style|generic>

<\body>
  <section|Information Theory>

  <subsection|Ensemble and Entropy>

  <\definition>
    [Ensemble] An ensemble <math|X> is a pair
    <math|<around*|(|A<rsub|X>,P<rsub|X>|)>> where
    <math|A<rsub|X>\<assign\><around*|{|x<rsub|1>,\<ldots\>,x<rsub|N>|}>> is
    alphabet and <math|P<rsub|X>\<assign\><around*|{|p<rsub|1>,\<ldots\>,p<rsub|N>|}>>
    are probabilities, s.t. <math|prob<around*|(|x<rsub|i>|)>=p<rsub|i>> and
    <math|<big|sum><rsub|i=1><rsup|N>p<rsub|i>=1>.
  </definition>

  <\definition>
    [Entropy of Ensemble] Let <math|X> an ensemble. Entropy of <math|X> is
    defined as

    <\equation*>
      H<around*|(|X|)>\<assign\>-<big|sum><rsub|i=1><rsup|N>p<rsub|i>log<rsub|2><around*|(|p<rsub|i>|)>
    </equation*>
  </definition>

  <subsection|Typical Set>

  <\definition>
    [Typical Set] Let <math|X> an ensemble. Given
    <math|N\<in\>\<bbb-Z\><rsub|+>> and <math|\<delta\>\<gtr\>0>, then the
    typical set of <math|X<rsup|N>> is defined as

    <\equation*>
      T<rsub|N\<delta\>>\<assign\><around*|{|x\<in\>A<rsub|X<rsup|N>>:prob<around*|(|<around*|\||<frac|1|N>log<rsub|2><frac|1|prob<around*|(|x|)>>-H<around*|(|X|)>|\|>\<less\>\<delta\>|)>|}>.
    </equation*>
  </definition>

  <\lemma>
    \;

    <\enumerate-numeric>
      <item>For <math|\<forall\>x\<in\>T<rsub|N\<delta\>>>,
      <math|2<rsup|-N<around*|(|H<around*|(|X|)>+\<delta\>|)>>\<less\>prob<around*|(|x|)>\<less\>><math|2<rsup|-N<around*|(|H<around*|(|X|)>-\<delta\>|)>>>.

      <item>For <math|\<forall\>\<epsilon\>\<gtr\>0,\<delta\>\<gtr\>0>,
      <math|\<exists\>N\<gtr\>0>, s.t. for <math|\<forall\>n\<gtr\>N>,
      <math|prob<around*|(|<around*|{|x\<in\>T<rsub|n\<delta\>>|}>|)>\<gtr\>1-\<epsilon\>>.
    </enumerate-numeric>
  </lemma>

  <\proof>
    Part one is straight forward. Now prove part two in the following.

    Notice that <math|prob<around*|(|x|)>=<big|prod><rsub|i=1><rsup|N>p<rsub|i>>,
    where <math|p<rsub|i>> is the probability of the <math|i>-th component of
    <math|x>. View <math|-log<rsub|2><around*|(|p<rsub|i>|)>> as random
    variable, being i.i.d. for all <math|i>, re-denoted by <math|Y<rsub|i>>.
    Thus by center limit theorem, the probability of
    <math|<wide|Y|\<bar\>>\<assign\><around*|(|1/N|)><big|sum><rsub|i=1><rsup|N>Y<rsub|i>>
    obeys normal distribution, with expectation <math|E<around*|(|Y|)>> and
    variance <math|Var<around*|(|Y|)>/<sqrt|N>>.

    Recall that

    <\equation*>
      E<around*|(|Y|)>=<big|sum><rsub|s>prob<around*|(|y<rsub|s>|)>y<rsub|s>=<big|sum><rsub|x<rsub|s>\<in\>A<rsub|X>>prob<around*|(|x<rsub|s>|)><around*|(|-log<rsub|2><around*|(|p<rsub|s>|)>|)>=H<around*|(|X|)>;
    </equation*>

    and

    <\equation*>
      Var<around*|(|Y|)>=<big|sum><rsub|s>prob<around*|(|y<rsub|s>|)><around*|(|y<rsub|s>-H<around*|(|X|)>|)><rsup|2>=<big|sum><rsub|x<rsub|s>\<in\>A<rsub|X>>prob<around*|(|x<rsub|s>|)><around*|(|-log<rsub|2><around*|(|p<rsub|s>|)>-H<around*|(|X|)>|)><rsup|2>
    </equation*>

    describing the expected derivation of <math|log<rsub|2><around*|(|p|)>>
    from <math|H<around*|(|X|)>>, being a finite constant, independent of
    <math|N>. Thus, the distribution of <math|<around*|(|1/N|)><big|sum><rsub|i=1><rsup|N>log<rsub|2><around*|(|1/p<rsub|i>|)>>,
    thus of <math|<around*|(|1/N|)>log<rsub|2><around*|(|1/prob<around*|(|x|)>|)>>,
    approximates a normal distribution with expectation
    <math|H<around*|(|X|)>> and variance propotional to <math|1/<sqrt|N>>.
    The part one is then proved.
  </proof>

  <\remark>
    [`Asymptotic Equipartition' Principle] We can say, without rigerousness,
    that almost all samples in <math|X<rsup|N>> is in the typical set
    <math|T<rsub|N\<delta\>>> for any given small <math|\<delta\>> as long as
    <math|N> is large enough. And all samples share the same probability
    <math|2<rsup|-NH<around*|(|X|)>>>.
  </remark>

  <subsection|The Source Coding Theorem>

  <\theorem>
    [Source Coding Theorem] Let <math|X> an ensemble. <math|X<rsup|N>> can be
    compressed into more than <math|NH<around*|(|X|)>> bits with negligible
    risk of information loss, as <math|N\<rightarrow\>\<infty\>>; conversely
    if they are compressed into fewer than <math|NH<around*|(|X|)>> bits it
    is virtually certain that information will be lost.
  </theorem>

  <\proof>
    If <math|N> is large enough, then almost all message is in the typical
    set of <math|X<rsup|N>>. There are <math|2<rsup|NH<around*|(|X|)>>>
    elements in typical set, being almost equal probability. Encoding
    <math|M> equal probability elements needs at least
    <math|log<rsub|2><around*|(|M|)>> bits, that is <math|NH<around*|(|X|)>>
    bits.
  </proof>

  <subsection|The Noisy-Channel Coding Theorem>

  \;
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|1.1|?>>
    <associate|auto-3|<tuple|1.2|?>>
    <associate|auto-4|<tuple|1.3|?>>
    <associate|auto-5|<tuple|1.4|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Information
      Theory> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <with|par-left|<quote|1tab>|1.1<space|2spc>Definitions
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <with|par-left|<quote|1tab>|1.2<space|2spc>The Source Coding Theorem
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|1tab>|1.3<space|2spc>The Noisy-Channel Coding
      Theorem <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>
    </associate>
  </collection>
</auxiliary>