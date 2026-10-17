# A degree and parity constraint for the 27-vertex B4/B9 search

Let \(G\) be a red graph on 27 vertices whose red edges avoid \(B_4\), and suppose the blue complement avoids \(B_9\). Let \(m=|E(G)|\), \(d_v\) be the red degrees, \(T\) the number of red triangles, and \(U\) the number of blue triangles. There are \(351-m\) blue edges.

Every red edge has at most three red common neighbors; every blue edge has at most eight blue common neighbors. Define the nonnegative page deficits

\[
D_R=3m-3T,\qquad D_B=8(351-m)-3U.
\]

Counting mixed triangles by their red/blue wedges gives

\[
T+U=\binom{27}{3}-26m+\tfrac12\sum_v d_v^2.
\]

For each vertex \(v\), the sum of red common-neighbor counts over its red incident edges equals twice the number of edges induced by its red neighborhood, hence is even. The red deficit incident to \(v\) has parity \(3d_v\), so it is at least one when \(d_v\) is odd. Each edge deficit is counted at both ends. Thus, with \(o\) the number of vertices of odd red degree,

\[
D_R\ge\frac{o}{2},\qquad D_R+D_B\ge\frac{o}{2}.
\]

Substituting the mixed-triangle identity yields the necessary inequality

\[
\boxed{3\sum_v d_v^2+o\le146m-11934.}
\]

For fixed \(m\), the left side is minimized by making the 27 degrees differ by at most one. Indeed, if \(a\le b-2\), replacing \((a,b)\) with \((a+1,b-1)\) reduces \(3(a^2+b^2)\) by at least six while increasing the odd-degree count by at most two. Using the balanced degrees for each integer \(m\) leaves exactly

\[
\boxed{157\le m\le171.}
\]

This interval is necessary, not sufficient. The calculation does not prove the existence or nonexistence of a 27-vertex witness. `search.py` implements the boxed inequality as `parity_condition_excess`, and the SAT search adds the edge-count interval. A dynamic program over all 27 degree choices from 0 to 26 independently confirmed the range 157–171.
