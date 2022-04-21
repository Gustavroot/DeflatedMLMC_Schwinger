# Some extra utils functions


import multigrid as mg

def flopsV(nr_levels, levels_info, level_id):
    #if nr_levels == 1:
    #    return levels_info[0].A.nnz
    #elif level_id == nr_levels-2:
    #    return 2 * levels_info[level_id].A.nnz + levels_info[level_id+1].A.nnz
    #else:
    #    return 2*levels_info[level_id].A.nnz + flopsV(nr_levels, levels_info, level_id+1)

    # adds the Krylov that wraps the AMG solver
    if nr_levels == 1:
        return levels_info[0].A.nnz
    elif level_id == nr_levels-2:
        #FIXME
        return (2+2) * levels_info[level_id].A.nnz + levels_info[level_id+1].A.nnz
    else:
        # FIXME
        if level_id==0:
            return (2+2) * levels_info[level_id].A.nnz + flopsV(nr_levels, levels_info, level_id+1)
        else:
            return (2+2) * levels_info[level_id].A.nnz + flopsV(nr_levels, levels_info, level_id+1)

# adds the Krylov that wraps the AMG solver
def flopsV_manual(bare_level, levels_info, level_id):
    #if bare_level==(len(levels_info)-1):
    #    return mg.coarsest_iters_avg * levels_info[0].A.nnz
    #elif level_id == nr_levels-2:
    if level_id == len(levels_info)-2:
        #return 2 * mg.smoother_iters * levels_info[level_id].A.nnz + mg.coarsest_iters_avg*levels_info[level_id+1].A.nnz
        # FIXME : number 30 hardcoded
        #return 2 * mg.smoother_iters * levels_info[level_id].A.nnz + 30*levels_info[level_id+1].A.nnz
        if level_id==bare_level:
            return (2 * mg.smoother_iters + 2) * levels_info[level_id].A.nnz + 0
        else:
            return (2 * mg.smoother_iters + 1) * levels_info[level_id].A.nnz + 0
    else:
        if level_id==bare_level:
            return (2 * mg.smoother_iters + 2) * levels_info[level_id].A.nnz + flopsV_manual(bare_level, levels_info, level_id+1)
        else:
            return (2 * mg.smoother_iters + 1) * levels_info[level_id].A.nnz + flopsV_manual(bare_level, levels_info, level_id+1)
