using  Graphs, GraphRecipes, Plots

g1 = Graph(3,2)

graphplot(g1)


edgelabels = Dict(
    (1,2) => 1,
    (1,3) => 2
)

function viewgraph(g)
    graphplot(
        g,
        names = 1:nv(g),
        fontsize = 8,
        nodeshape = :circle,
        marksize = 0.15,
        markerstrokewidth = 2,
        edgelabels = edgelabels,
        linewidth = 1,
        curves = false

    )
end

p1 = viewgraph(g1)          

gh = Graph()

add_vertices!(gh,5)#그래프와 노드의 개수 1~~N

add_edge!(gh,2,3)# 노드와 노드 연결 번호

p2 = viewgraph(gh)  

savegraph("graph1.lgz",gh)  


g3 = loadgraph("graph1.lgz")
p3 = viewgraph(gh)