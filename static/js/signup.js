const signupForm = document.getElementById("signupForm");
const signupMessage = document.getElementById("signupMessage");

signupForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {
    name: document.getElementById("signupName").value.trim(),
    birthDate: document.getElementById("signupBirthDate").value,
    username: document.getElementById("signupUsername").value.trim(),
    password: document.getElementById("signupPassword").value.trim()
  };

  try {
    const response = await fetch("/api/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.message || "회원가입에 실패했습니다.");
    }

    signupMessage.textContent = result.message;
    signupMessage.className = "auth-message success";
    alert("회원가입이 완료되었습니다.");
    signupForm.reset();

    if (window.opener && !window.opener.closed) {
      window.opener.focus();
    }

    window.close();
  } catch (error) {
    signupMessage.textContent = error.message;
    signupMessage.className = "auth-message error";
  }
});
