"""Compact bilingual-ready tables and a Chinese report from verified experiment outputs."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path('/Users/sonny/Desktop/OCO2/outputs/million_pc_sensitivity_20261024')
LABEL={'GL_OCEAN':'海洋 GL','GL_LAND':'陆地 GL','ND':'ND'}


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
        '| '+' | '.join(['---']*len(headers))+' |',
        *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def main():
    verification=json.loads((ROOT/'independent_verification.json').read_text())
    assert verification['status']=='passed'
    d=pd.read_csv(ROOT/'metrics.csv')
    assert len(d)==336
    frozen=json.loads((ROOT/'selection_frozen.json').read_text())
    keys=['domain','recipe','k']
    annual=d.groupby(keys+['year','role']).agg(
        rmse_mean=('rmse','mean'),rmse_seed_sd=('rmse','std'),
        bias_mean=('bias','mean'),bias_seed_sd=('bias','std'),
        r2_mean=('r2','mean'),r2_seed_sd=('r2','std'),n=('n','first'),
        lr=('lr','first')).reset_index()
    annual.to_csv(ROOT/'annual_metrics_summary.csv',index=False)
    per_seed=d[d.year>2020].groupby(keys+['seed'])[['rmse','bias','r2']].mean()
    test=per_seed.groupby(keys).agg(test_rmse_mean=('rmse','mean'),
        test_rmse_seed_sd=('rmse','std'),test_bias_mean=('bias','mean'),
        test_r2_mean=('r2','mean')).reset_index()
    train=annual[annual.year==0][keys+['rmse_mean']].rename(columns={'rmse_mean':'train_rmse'})
    val=annual[annual.year==2020][keys+['rmse_mean','rmse_seed_sd','bias_mean','r2_mean','lr']].rename(
        columns={'rmse_mean':'validation_rmse','rmse_seed_sd':'validation_seed_sd',
                 'bias_mean':'validation_bias','r2_mean':'validation_r2'})
    summary=train.merge(val,on=keys).merge(test,on=keys)
    summary.to_csv(ROOT/'comparison_summary.csv',index=False)
    epochs=pd.DataFrame(verification['audit'])
    epochs.to_csv(ROOT/'selected_epoch_audit.csv',index=False)
    report=['# 百万训练样本 PC 敏感性实验：已完成独立核验',
        '\n## 最重要的结论',
        '\n**十万样本下“20 PC 的验证表现最好”的结论不能直接推广到百万样本。** '
        '百万样本下，三个域的固定配方均由 30 PC 取得最低的 2020 验证 RMSE；'
        'ND 在逐配置筛选学习率后，也是 30 PC 验证最优。',
        '\n但 ND 的 2021–2025 年平均年度测试误差仍是 20 PC 较低。'
        '这说明“训练后紧邻年份的验证最优”和“后续多年迁移最优”并不一致。'
        '不能利用已经查看的未来测试表现，反过来把 20 PC 宣称为按验证集选出的主模型。',
        '\n## 1. 固定主模型配方的百万样本对照',
        '\n每个域均使用 2015–2019 年全部 1,000,000 条训练样本及 2020 年全部 '
        '200,000 条验证样本。0 PC 仍含九个辅助及先验输入。以下 PC 数为每波段数量。'
        'RMSE 单位 ppm，均为三个种子均值。测试均值先在每个种子内对五个年度 '
        'RMSE 等权平均，再跨种子汇总，不是合并观测后的 RMSE。',
        table(['域','每波段 PC','训练 RMSE','2020 验证 RMSE','平均年度测试 RMSE'],
            [[LABEL[r.domain],r.k,f'{r.train_rmse:.3f}',f'{r.validation_rmse:.3f}',
              f'{r.test_rmse_mean:.3f} ± {r.test_rmse_seed_sd:.3f}']
             for r in summary[summary.recipe=='fixed'].itertuples()]),
        '\n± 为三个种子的标准差，不是置信区间。海洋 GL 的 20 与 30 PC 多年误差接近；'
        '陆地 GL 的 30 PC 验证和多年误差均较低；ND 的排序则随验证期和后续年份而变化。',
        '\n## 2. ND 公平学习率筛选',
        '\n每个 PC 配置都筛选原学习率、1e-4、3e-4 和 1e-3；种子 17 初筛前两名，'
        '再用 17、29、43 复核，按平均完整 2020 验证 RMSE 冻结学习率。'
        '网络、数据、批量、dropout、500 轮上限及早停规则不变。',
        table(['每波段 PC','选中学习率','训练 RMSE','验证 RMSE','平均年度测试 RMSE'],
            [[r.k,f'{r.lr:.6g}',f'{r.train_rmse:.3f}',f'{r.validation_rmse:.3f} ± {r.validation_seed_sd:.3f}',
              f'{r.test_rmse_mean:.3f} ± {r.test_rmse_seed_sd:.3f}']
             for r in summary[summary.recipe=='lr_selected'].itertuples()]),
        '\n30 PC 仍选择原学习率约 3.814e-5，而不是十万样本实验中选出的 1e-3。'
        '同样轮数下，百万样本每轮包含约十倍参数更新，因此小样本最优学习率也不能直接照搬。'
        '20 PC 提高学习率改善了 2020 验证表现，但其多年测试均值从 1.322 变为 1.359 ppm，'
        '提示只优化 2020 验证表现并不保证后续年份也改善。',
        '\n## 3. ND 20 与 30 PC 的年度差异（固定配方）',
        table(['年份','n / 种子','20 PC RMSE','20 PC bias','20 PC R²',
               '30 PC RMSE','30 PC bias','30 PC R²'],
            [[str(year)+('*' if year==2025 else ''),int(a.n),f'{a.rmse_mean:.3f}',
              f'{a.bias_mean:+.3f}',f'{a.r2_mean:.3f}',f'{b.rmse_mean:.3f}',
              f'{b.bias_mean:+.3f}',f'{b.r2_mean:.3f}']
             for year in range(2020,2026)
             for a in [annual[(annual.domain=='ND')&(annual.recipe=='fixed')&(annual.k==20)&(annual.year==year)].iloc[0]]
             for b in [annual[(annual.domain=='ND')&(annual.recipe=='fixed')&(annual.k==30)&(annual.year==year)].iloc[0]]]),
        '\n2020 为验证期；2021–2025 为回溯测试，2025* 是部分年度。bias 定义为预测减 ACOS 参考值。'
        '30 PC 在 2020–2021 年误差更低，后续正偏增大；20 PC 的后期偏差较小。'
        '这支持进一步研究时间迁移，但不能单凭该结果确定是高阶 PC、仪器变化或参考产品版本造成。',
        '\n## 4. 轮数、可比性与核验',
        f'\n48 条最终比较模型记录中，{len(verification["cap500_records"])} 条达到 500 轮上限。'
        '若同一检查点同时出现在固定和学习率选优比较中，不视为两次独立拟合。'
        '逐种子的最佳/停止轮数、训练/验证误差见 selected_epoch_audit.csv。',
        '\n陆地 GL 原模型采用 200 轮余弦计划。本次所有陆地 GL 配置（包括 30 PC）'
        '统一采用 500 轮余弦计划重新训练，保证本次内部对照一致；因此其 30 PC '
        '不是原论文模型的完全重复。海洋 GL 与 ND 的恒定学习率 30 PC 检查点直接复用，'
        '并已核验其年度结果与原百万样本主模型一致。',
        '\n独立核验通过：48 条模型记录、336 条指标、288 个年度预测文件；'
        '训练及验证预测从保存模型重新计算；训练专用标准化参数从训练数据重新计算；'
        '全部预测的 sounding_id、original_index、参考值及先验逐元素与源域/年份匹配；'
        '模式标签、样本量、训练历史和学习率计划正确，所有指标有限。',
        '\n固定网络原先在 30 PC 下选出，本次并未针对每个 k 重搜网络，'
        '因此这是该架构下的敏感性分析，不是所有架构和 PC 数的全局最优搜索。'
        '所有测试年份在以前研究中已被查看，不能称为新设盲测。'
        '各配置只有三个种子，误差差异不能据均值直接认定为统计显著。',
        '\n## 5. 对论文的建议',
        '\n按既定 2020 验证准则，主实验保留每波段 30 PC 有依据。'
        '可以把百万样本敏感性作为补充结果，并如实讨论 ND 中验证排序与多年测试排序不一致。'
        '若要专门优化跨年迁移，应预先设计仅在训练/开发年份内的滚动验证，'
        '而不是反复依据 2021–2025 测试结果调参。现有中英文稿及主结果尚未替换。',
        '\n完整逐域/年份 RMSE、bias、R²、样本数和种子标准差见 annual_metrics_summary.csv；'
        '汇总见 comparison_summary.csv；核验记录见 independent_verification.json。']
    (ROOT/'RESULTS_CN.md').write_text('\n\n'.join(report)+'\n')
    print(summary.to_string(index=False))
    print('CAP500',len(verification['cap500_records']))


if __name__=='__main__':
    main()
