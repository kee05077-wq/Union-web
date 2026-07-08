const findIdForm = document.getElementById("findIdForm");
const findPasswordForm = document.getElementById("findPasswordForm");

findIdForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {
    name: document.getElementById("findIdName").value.trim(),
    birthDate: document.getElementById("findIdBirthDate").value
  };

  try {
    const response = await fetch("/api/find-id", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.message || "아이디를 찾지 못했습니다.");
    }

    alert(`회원님의 아이디는 "${result.username}" 입니다.`);
    findIdForm.reset();
  } catch (error) {
    alert(error.message);
  }
});

findPasswordForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {
    username: document.getElementById("findPasswordUsername").value.trim(),
    name: document.getElementById("findPasswordName").value.trim(),
    birthDate: document.getElementById("findPasswordBirthDate").value
  };

  try {
    const response = await fetch("/api/find-password", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.message || "비밀번호를 찾지 못했습니다.");
    }

    alert(`회원님의 비밀번호는 "${result.password}" 입니다.`);
    findPasswordForm.reset();
  } catch (error) {
    alert(error.message);
  }
});
