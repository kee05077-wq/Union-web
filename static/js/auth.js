const authButtons = document.querySelectorAll(".btn-auth");
const authModal = document.getElementById("authModal");
const authClose = document.getElementById("closeAuthModal");
const loginForm = document.getElementById("loginForm");
const loginMessage = document.getElementById("loginMessage");
const openRecoverButton = document.getElementById("openRecoverWindow");
const openSignupButton = document.getElementById("openSignupWindow");

function openAuthModal(event) {
  if (event) {
    event.preventDefault();
  }
  authModal.classList.remove("auth-hidden");
}

function closeAuthModal() {
  authModal.classList.add("auth-hidden");
  loginMessage.textContent = "";
  loginMessage.className = "auth-message";
}

function openPopup(url, name) {
  window.open(url, name, "width=760,height=880,scrollbars=yes,resizable=yes");
}

authButtons.forEach((button) => {
  button.addEventListener("click", openAuthModal);
});

if (authClose) {
  authClose.addEventListener("click", closeAuthModal);
}

if (authModal) {
  authModal.addEventListener("click", (event) => {
    if (event.target === authModal) {
      closeAuthModal();
    }
  });
}

if (openRecoverButton) {
  openRecoverButton.addEventListener("click", () => {
    openPopup("/recover.html", "unionRecoverWindow");
  });
}

if (openSignupButton) {
  openSignupButton.addEventListener("click", () => {
    openPopup("/signup.html", "unionSignupWindow");
  });
}

if (loginForm) {
  loginForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const payload = {
      username: document.getElementById("loginUsername").value.trim(),
      password: document.getElementById("loginPassword").value.trim()
    };

    try {
      const response = await fetch("/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.message || "로그인에 실패했습니다.");
      }

      loginMessage.textContent = `${result.message} (${result.profile.username})`;
      loginMessage.className = "auth-message success";
      alert(`${result.profile.name}님의 로그인에 성공했습니다.`);
      loginForm.reset();
      closeAuthModal();
    } catch (error) {
      loginMessage.textContent = error.message;
      loginMessage.className = "auth-message error";
    }
  });
}
