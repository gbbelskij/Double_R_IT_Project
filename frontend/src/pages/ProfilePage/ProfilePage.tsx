import Footer from "@components/Footer/Footer";
import Header from "@components/Header/Header";
import Logo from "@components/Logo/Logo";
import Main from "@components/Main/Main";

import "./ProfilePage.css";

const ProfilePage: React.FC = () => {
  return (
    <>
      <Header />
      <Main>
        <Logo hasText />
      </Main>
      <Footer />
    </>
  );
};

export default ProfilePage;
