
__kernel void vector_add(__constant int *agent, __constant int *G,
                                  __global float *C, __constant float *std2,
                                  __constant int *GENES) {
    
    // Get the index of the current element
    int i = get_global_id(0);
    int curAgent = i / (*GENES);
    // int gene = i - (curAgent * *GENES); // current geneid 
    int gene = i % *GENES;
    // Do the operation
    int4 tmp = vload4(gene, agent);
    // int4 tmp = (int4)(agent[gene], agent[gene+1], agent[gene+2], agent[gene+3]);
    tmp -= vload4((curAgent * *GENES) + gene, G);
    // tmp *= tmp;
    float4 tmp2 = convert_float4_rtp(tmp);
    tmp2 *= tmp2;
    tmp2 /= vload4(gene, std2);

    if (tmp2.w != tmp2.w) tmp2.w = 0.0; 
    if (tmp2.x != tmp2.x) tmp2.x = 0.0; 
    if (tmp2.y != tmp2.y) tmp2.y = 0.0; 
    if (tmp2.z != tmp2.z) tmp2.z = 0.0; 
    // C[curAgent] += (float)(tmp.s0 + tmp.s1 + tmp.s2 + tmp.s3);
    C[curAgent] += (float)(tmp2.s0 + tmp2.s1 + tmp2.s2 + tmp2.s3);


    
}

