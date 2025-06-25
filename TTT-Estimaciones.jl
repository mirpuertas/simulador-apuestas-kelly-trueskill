using CSV
using Dates
using DataFrames
using TrueSkillThroughTime
global const ttt = TrueSkillThroughTime


df_atp = CSV.read("Trabajo-Final/Final-Apuestas/data/df_wta.csv", DataFrame,stringtype=String)
df_atp = df_atp[.!ismissing.(df_atp.Date) .&& .!ismissing.(df_atp.B365W) .&& .!ismissing.(df_atp.B365L), :]
df_atp = sort(df_atp, :Date)


times = [ Dates.value(row["Date"]-Date("1900-01-01"))  for row in eachrow(df_atp)]

composition = [ [[row["Winner"]],[row["Loser"]]]  for row in eachrow(df_atp)]

h = ttt.History(composition=composition, times = times, online=true, iterations=1, sigma=2.5, gamma=0.036)

df = DataFrame()
df.winner = [ ev.teams[1].items[1].agent for b in h.batches for ev in b.events]
df.loser = [ ev.teams[2].items[1].agent for b in h.batches for ev in b.events]
df.m_winner = [ b.skills[ev.teams[1].items[1].agent].online.mu for b in h.batches for ev in b.events]
df.m_loser = [ b.skills[ev.teams[2].items[1].agent].online.mu for b in h.batches for ev in b.events]
df.s_winner = [ b.skills[ev.teams[1].items[1].agent].online.sigma for b in h.batches for ev in b.events]
df.s_loser = [ b.skills[ev.teams[2].items[1].agent].online.sigma for b in h.batches for ev in b.events]

df.b_winner = df_atp.B365W # Bet365
df.b_loser = df_atp.B365L # Bet365
df.date     = df_atp.Date

CSV.write("Trabajo-Final/Final-Apuestas/inferencia/predicciones_wta.csv", df; header=true, encoding="UTF-8")