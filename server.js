const express = require("express");
const fs = require("fs");
const path = require("path");
const XLSX = require("xlsx");

const app = express();
const PORT = 3000;

const rootDir = __dirname;
const databaseDir = path.join(rootDir, "database");
const workbookPath = path.join(databaseDir, "users.xlsx");
const sheetName = "Users";

function ensureWorkbook() {
  if (!fs.existsSync(databaseDir)) {
    fs.mkdirSync(databaseDir, { recursive: true });
  }

  if (!fs.existsSync(workbookPath)) {
    const workbook = XLSX.utils.book_new();
    const worksheet = XLSX.utils.json_to_sheet([]);
    XLSX.utils.book_append_sheet(workbook, worksheet, sheetName);
    XLSX.writeFile(workbook, workbookPath);
  }
}

function readUsers() {
  ensureWorkbook();
  const workbook = XLSX.readFile(workbookPath);
  const worksheet = workbook.Sheets[sheetName];
  if (!worksheet) {
    return [];
  }
  return XLSX.utils.sheet_to_json(worksheet, { defval: "" });
}

function writeUsers(users) {
  ensureWorkbook();
  const workbook = XLSX.utils.book_new();
  const worksheet = XLSX.utils.json_to_sheet(users);
  XLSX.utils.book_append_sheet(workbook, worksheet, sheetName);
  XLSX.writeFile(workbook, workbookPath);
}

function sameText(a, b) {
  return String(a || "").trim() === String(b || "").trim();
}

app.use(express.json());
app.use(express.static(rootDir));

app.get("/", (req, res) => {
  res.sendFile(path.join(rootDir, "index.html"));
});

app.post("/api/signup", (req, res) => {
  const { name, birthDate, username, password } = req.body || {};

  if (!name || !birthDate || !username || !password) {
    return res.status(400).json({ message: "모든 항목을 입력해 주세요." });
  }

  const users = readUsers();
  const duplicate = users.some((user) => sameText(user.username, username));

  if (duplicate) {
    return res.status(409).json({ message: "이미 사용 중인 아이디입니다." });
  }

  users.push({
    name: String(name).trim(),
    birthDate: String(birthDate).trim(),
    username: String(username).trim(),
    password: String(password).trim(),
    createdAt: new Date().toISOString()
  });

  writeUsers(users);
  return res.json({ message: "회원가입이 완료되었습니다." });
});

app.post("/api/login", (req, res) => {
  const { username, password } = req.body || {};
  const users = readUsers();
  const user = users.find(
    (item) => sameText(item.username, username) && sameText(item.password, password)
  );

  if (!user) {
    return res.status(401).json({ message: "아이디 또는 비밀번호가 일치하지 않습니다." });
  }

  return res.json({
    message: `${user.name}님, 로그인되었습니다.`,
    profile: {
      name: user.name,
      birthDate: user.birthDate,
      username: user.username
    }
  });
});

app.post("/api/find-id", (req, res) => {
  const { name, birthDate } = req.body || {};
  const users = readUsers();
  const user = users.find(
    (item) => sameText(item.name, name) && sameText(item.birthDate, birthDate)
  );

  if (!user) {
    return res.status(404).json({ message: "일치하는 회원정보를 찾지 못했습니다." });
  }

  return res.json({ username: user.username });
});

app.post("/api/find-password", (req, res) => {
  const { username, name, birthDate } = req.body || {};
  const users = readUsers();
  const user = users.find(
    (item) =>
      sameText(item.username, username) &&
      sameText(item.name, name) &&
      sameText(item.birthDate, birthDate)
  );

  if (!user) {
    return res.status(404).json({ message: "입력하신 정보와 일치하는 비밀번호를 찾지 못했습니다." });
  }

  return res.json({ password: user.password });
});

app.listen(PORT, () => {
  ensureWorkbook();
  console.log(`Union-web server running at http://localhost:${PORT}`);
});
