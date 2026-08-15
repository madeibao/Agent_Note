
---
name: roll-dice
description: 使用随机数生成器掷骰子。当用户要求掷骰子（d6、d20 等）或生成随机骰子点数时使用。
---

# 掷骰子方法

使用以下 shell 命令生成 1 到指定面数之间的随机数：

## macOS / Linux

```bash
echo $((RANDOM % <sides> + 1))
```

## Windows PowerShell

```powershell
Get-Random -Minimum 1 -Maximum (<sides> + 1)
```

**用法说明：**
将 `<sides>` 替换为用户指定的骰子面数：
- d6：将 `<sides>` 替换为 6
- d20：将 `<sides>` 替换为 20
- 其他面数以此类推

**输出：** 直接返回随机结果数字即可，无需额外解释。