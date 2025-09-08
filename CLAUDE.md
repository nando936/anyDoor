# AnyDoor Project Instructions

## Project Overview
AnyDoor - Door monitoring and control project

## Order Processing Instructions

### WHEN USER SAYS "PROCESS ORDER":
1. **FORGET EVERYTHING** - Act as if you've NEVER seen these files before
2. **NO "I ALREADY..."** - Never say "I already ran" or "I already extracted"
3. **START AT STEP 1** - Even if you think you did it before
4. **READ** `order_processing/README.md` - This is the GOVERNING DOCUMENT
5. **FOLLOW** those instructions from the beginning

### KEY REMINDERS (Full details in README.md):
- **Manual extraction** = Claude reads PDF and fills template (not scripts)
- **No new scripts** = Use existing tools only
- **Start fresh** = Never use old data or templates
- **Three-step rule**: Extract → Process → Verify

### ⚠️ RUSH CHECK:
**SLOW IS FAST** - Following README.md correctly first time > restarting 3 times
- Creating scripts? STOP - you're rushing
- Automating extraction? STOP - you're rushing  
- Skipping steps? STOP - you're rushing

### 📍 WHERE TO UPDATE WHEN ISSUES ARISE:
- **Order processing workflow issues** → Update `order_processing/README.md`
- **Claude-specific behavior issues** → Update this file (CLAUDE.md)
- **Technical/code issues** → Fix the actual scripts

**GOVERNING DOCUMENT:** `order_processing/README.md` - Always defer to it for order processing

## Quick Commands
- `*` - Check comments.txt for tasks and process them

## Comments.txt File Rules
**LOCATION**: comments.txt is ALWAYS in the PROJECT ROOT FOLDER
- For anydoor project: C:\Users\nando\Projects\anydoor\comments.txt

See `comments_text_file__execution_guide.txt` in project root for all comments.txt instructions and usage.

**IMPORTANT**: When handling comments:
- Comments from webpage should be marked as: ### nando Comment (Webpage) - HH:MM
- Comments from CLI should be marked as: ### nando Comment (CLI) - HH:MM
- Claude responses should be marked as: ### Claude Response (CLI)
- Follow the execution guide for processing and responding to comments
- **ONLY respond to comments that are already in comments.txt** - Do not post chat conversations to comments.txt
- When using the `*` command, read and respond to existing comments in the file

**CRITICAL - NO SPECIAL CHARACTERS IN COMMENTS**:
- **NEVER use emojis** (no emojis of any kind)
- **Use plain text only** - no special Unicode characters
- **Use standard ASCII characters** for formatting:
  - Use [x] instead of checkmarks
  - Use [!] instead of warning symbols
  - Use * or - for bullet points
  - Use ** for bold text
- This ensures comments display properly on the webpage without JavaScript errors

**CRITICAL - 40 CHARACTER WIDTH**:
- **Max 40 characters per line**
- Comments are read on phone screens
- Break long lines into multiple lines
- Format lists to be narrow
- Keep text compact for mobile viewing

## Session Startup
1. Check comments.txt for any pending tasks
2. Review project status
3. Continue with development