# A necessary condition for a 12-regular 27-vertex witness

Suppose a 12-regular graph \(G\) on 27 vertices avoids red \(B_4\) and its complement avoids blue \(B_9\). Write \(A\) for its red adjacency matrix and \(c(u,v)=|N_G(u)\cap N_G(v)|=(A^2)_{uv}\).

For a red edge, \(c(u,v)\le3\). For a nonedge, its blue common-neighbor count is \(25-12-12+c(u,v)=1+c(u,v)\), so \(c(u,v)\le7\).

Define the symmetric integer matrix

\[
D=5I-4A+7J-A^2.
\]

Its diagonal is zero. On a red edge, \(D_{uv}=3-c(u,v)\ge0\); on a nonedge, \(D_{uv}=7-c(u,v)\ge0\). Every row sums to

\[
5-4(12)+7(27)-12^2=2.
\]

Thus \(D\) is the adjacency matrix of a **2-regular multigraph**: a union of cycles, allowing a double edge as a two-vertex component. Equivalently, each vertex has exactly two units of deficit from the red-edge codegree cap 3 and nonedge codegree cap 7. The total deficit over unordered vertex pairs is 27.

This applies only to the 12-regular slice. Other degree sequences remain possible under the general necessary condition. It is a derived search constraint, not a proof that a witness exists or that it cannot exist.
